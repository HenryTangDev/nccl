# NCCL Code Generation — Q&A

Grounded in `src/device/generate.py`, `src/include/device.h`, and `src/device/op128.h`.

---

## How does `best_kernel()` choose kernel variants?

`generate.py:147-158` — it maps every (coll, redop, type, algo, proto) tuple to the **one kernel that will be compiled as a specialized `__global__` entry point** for that tuple:

```python
def best_kernel(coll, redop, ty, algo, proto):
  def best(coll, redop, ty, algo, proto):
    if coll == "Nop":     return ("Generic", None, None, None, None)
    if coll == "SendRecv": return ("SendRecv", None, None, None, None)
    if coll in ("AllGather","Broadcast","AllGatherV"):
      return (coll, None, None, "RING", "LL")            # ← always RING/LL kernel
    return (coll, "Sum", ty, ("TREE" if algo=="TREE" else "RING"), "LL")  # ← Sum/LL
  kfn = equivalent_primary(*best(coll, redop, ty, algo, proto))
  if not func_filter(*kfn): return ("Generic", None, None, None, None)
  return kfn
```

The logic is deliberately narrow:

| Collective | Specialized kernel chosen |
|---|---|
| `SendRecv` | One kernel, no variants |
| `AllGather`, `Broadcast` | `(coll, RING, LL)` — one kernel regardless of algo/proto |
| `AllReduce`, `Reduce`, `ReduceScatter` | `(coll, Sum, <type>, RING or TREE, LL)` — per-type, Sum only, LL proto |

**What this means in practice**: only a small fraction of (algo, proto, redop) combinations get their own `__global__` kernel. All other combinations share the nearest kernel and call the correct `ncclDevFunc_*` *device function* through a function pointer table — paying one extra indirect call, but saving thousands of kernel binaries.

The result is two generated tables in `host_table.cc`:
- `ncclDevKernelForFunc[]` — maps every primary function id → kernel pointer
- `ncclDevKernelForFuncIsSpecialized[]` — `1` if `fn == kfn` (it got its own kernel), `0` if it's sharing

---

## What is an equivalence class of kernels?

`generate.py:134-142` — `equivalent_primary()` maps certain (redop, type) pairs to a single **representative** that implements them identically:

```python
def equivalent_primary(coll, redop, ty, algo, proto):
  if coll in ("AllReduce", "Reduce", "ReduceScatter"):
    # signed int sum/prod/PreMulSum/SumPostDiv → use unsigned version
    if redop in ("Sum","Prod","PreMulSum","SumPostDiv") and ty[0] == "i":
      return (coll, redop, "u"+ty[1:], algo, proto)   # i32 → u32, i64 → u64
    # signed int MinMax → use unsigned (except NVLS which needs native signed ops)
    if redop == "MinMax" and ty[0] == "i" and "NVLS" not in algo:
      return (coll, redop, "u"+ty[1:], algo, proto)
  return (coll, redop, ty, algo, proto)               # everything else: is its own primary
```

**Why signed → unsigned works**: for `Sum` and `Prod`, two's complement arithmetic is identical for signed and unsigned integers — `0xFF + 0x01` gives `0x00` whether you call it `i8` or `u8`. Same for `MinMax` in non-NVLS paths (where the comparison uses unsigned bit patterns which match for same-magnitude integers in NCCL's usage). NVLS uses hardware reduction units that have native signed/unsigned distinction, so those stay separate.

The practical effect: instead of compiling 12 type variants × 5 redops = 60 combinations, many collapse:

```
(AllReduce, Sum, i8,  RING, LL)  → (AllReduce, Sum, u8,  RING, LL)  ← same object code
(AllReduce, Sum, i32, RING, LL)  → (AllReduce, Sum, u32, RING, LL)  ← same object code
(AllReduce, Sum, i64, RING, LL)  → (AllReduce, Sum, u64, RING, LL)  ← same object code
```

The total set of unique implementations — `primary_funcs` — is the de-duplicated output after running every candidate through `equivalent_primary()`.

---

## How does `BytePack<16>` achieve 128-bit memory access?

`src/device/op128.h:132-156` — `BytePack<16>` is a `union` with 16-byte alignment, and its load/store specializations emit PTX `v2.b64` (two 64-bit) instructions:

```cpp
// op128.h:132
template<>
union alignas(16) BytePack<16> {
  BytePack<8> half[2];
  uint8_t  u8[16];
  uint16_t u16[8];
  uint32_t u32[4];
  uint64_t u64[2];
  ulong2   ul2[1], native;   // ← ulong2 is CUDA's 128-bit type
};
```

The 16-byte load/store specializations (`op128.h:267-286`) emit a single `v2.b64` PTX instruction:

```cpp
// ld_global<16> specialization:
asm volatile("ld.global.v2.b64 {%0,%1}, [%2];"
    : "=l"(ans.u64[0]), "=l"(ans.u64[1]) : "l"(addr) : "memory");

// st_global<16> specialization:
asm volatile("st.global.v2.b64 [%0], {%1,%2};"
    :: "l"(addr), "l"(value.u64[0]), "l"(value.u64[1]) : "memory");
```

`v2.b64` is a PTX *vector* instruction — the GPU issues a **single 128-bit memory transaction** loading both 64-bit halves atomically in one L2 cache line access. This is the maximum width a single CUDA thread can transfer per instruction, matching one full L2 cache line sector (128 bits = one sector of a 128-byte L2 line).

The `reduceCopyPacks()` inner loop in `common_kernel.h` uses `BytePack<N>` as its template parameter to control vectorization width:

```
BytePack<16>: one v2.b64 → 128 bits / thread / instruction  ← maximum bandwidth
BytePack<8>:  one b64    →  64 bits / thread / instruction
BytePack<4>:  one b32    →  32 bits / thread / instruction
```

With `Unroll=8` and a full warp (32 threads), one loop iteration of `reduceCopyPacks<..., BytePack<16>>` moves `8 × 32 × 16 = 4096 bytes` — one complete cache line per thread per unrolled iteration.

### BytePack size hierarchy

```
BytePack<16>  alignas(16)
  └─ half[2]:  BytePack<8>
       └─ half[2]: BytePack<4>
            └─ half[2]: BytePack<2>
                 └─ half[2]: BytePack<1>
```

Each level carries union views at all smaller granularities (`u8[]`, `u16[]`, `u32[]`, `u64[]`), so reduction code can reinterpret the same 16 bytes as any element type without casts.

---

## Why must enum order in `device.h` match `generate.py`?

The enums in `device.h` are used as **integer indices** directly in the `ncclDevFuncId()` arithmetic formula. `generate.py` walks its own Python lists in the same order to produce the `ncclDevFuncRowToId[]` array. If the two orderings diverge, the same integer index resolves to different functions in each side — no compile error, just silent wrong-kernel dispatch at runtime.

### The formula (device.h:582)

```c
// Comment in device.h:581 says explicitly:
// "`ncclDevFuncIndex()` needs to be in sync with all_functions() in generate.py"

inline int ncclDevFuncId(int coll, int devRedOp, int type, int algo, int proto) {
  // For AllReduce:
  row += ((devRedOp * NumTypes + type) * nAlgos + algo) * NCCL_NUM_PROTOCOLS + proto;
  return ncclDevFuncRowToId[row];
}
```

`devRedOp`, `type`, `algo`, `proto` are all **raw enum integer values** used as multipliers in this formula. The formula hard-codes the sizes (`NumTypes`, `nAlgos`, `NCCL_NUM_PROTOCOLS`).

### The Python side (generate.py:6-11, 161-174)

```python
all_redops = ["Sum","Prod","MinMax","PreMulSum","SumPostDiv"]   # ← index 0,1,2,3,4
all_tys    = ["i8","u8","i32","u32","i64","u64","f16","f32","f64","bf16","f8e4m3","f8e5m2"]
all_protos = ["LL","LL128","SIMPLE"]
all_algos  = ["TREE","RING","COLLNET_DIRECT","COLLNET_CHAIN","NVLS","NVLS_TREE","PAT"]

def enumerate_func_rows():
  yield ("SendRecv", ...)          # row 0
  for coll in ("AllGather", ...):  # rows 1..N
    for algo in algos_of_coll[coll]:
      for proto in all_protos:
        yield (coll, None, None, algo, proto)
  for coll in ("AllReduce", ...):
    for redop in all_redops:        # outer loop → redop is high dimension
      for ty in all_tys:
        for algo in algos_of_coll[coll]:
          for proto in all_protos:  # inner loop → proto is low dimension
            yield (coll, redop, ty, algo, proto)
```

The Python list position (`all_redops.index("Sum") == 0`) must equal the C enum value (`ncclDevSum == 0`). The nested loop order must match the formula's multiplier order.

### What breaks if they diverge

```
Example: if generate.py lists redops as ["Prod","Sum",...] but device.h has
         ncclDevSum=0, ncclDevProd=1

ncclDevFuncId(AllReduce, ncclDevSum=0, f32, RING, LL)
  → row = (0 * 12 + ...) * 6 * 3 = points to row for "Prod/f32/RING/LL"
                                                         ^^^^ WRONG KERNEL
```

The generated `ncclDevFuncRowToId[]` array has Prod at index 0 (Python order), but the C formula uses `ncclDevSum=0` as if it were index 0 — so it dispatches the Prod kernel for a Sum operation. The GPU executes the wrong reduction silently.

### The four affected enum groups (`generate.py:6, device.h` comment on line 6)

```
generate.py comment:  "Order of redops, tys, protos, algos must match src/include/device.h"

all_redops  ↔  enum ncclDevRedOp_t  { ncclDevSum=0, ncclDevProd=1, ncclDevMinMax=2, ... }
all_tys     ↔  enum ncclDataType_t  { ncclInt8=0, ncclUint8=1, ncclInt32=2, ... }
all_protos  ↔  #define NCCL_PROTO_LL=0, NCCL_PROTO_LL128=1, NCCL_PROTO_SIMPLE=2
all_algos   ↔  #define NCCL_ALGO_TREE=0, NCCL_ALGO_RING=1, NCCL_ALGO_COLLNET_DIRECT=2, ...
```

Collectives are handled separately — the `ncclDevFuncId()` formula uses explicit `if (coll == ncclFuncAllReduce)` branches, not a multiplier, so coll enum order only matters within those branches.
