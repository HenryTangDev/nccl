# NCCL Deep Dive Learning Guide

A practical, layered reading plan. Each stage builds on the previous —
don't jump ahead until the current stage feels solid.

---

## Stage 1: What NCCL Does (1–2 days)

**Goal**: Understand the problem space and API surface before touching internals.

1. **Read the public header** `build/include/nccl.h` (or `src/nccl.h.in`)
   - Every collective, every enum, every config option
   - Notice: operations are always async, tied to a `cudaStream_t`

2. **Run the examples** in `examples/` in order (01 → 06)
   - Build with `make -j src.build` then compile examples
   - `01_communicators` → `03_collectives` are the must-reads

3. **Key questions to answer**: What is a *communicator*? What is a *rank*? What is a *channel*?

---

## Stage 2: One Collective End-to-End (3–5 days)

**Goal**: Trace a single `ncclAllReduce` call from API to GPU kernel.

Follow this chain in order:

```
src/collectives.cc          ← ncclAllReduce() creates ncclInfo struct
src/enqueue.cc              ← ncclEnqueueCheck() validates + schedules
src/enqueue.cc              ← algorithm/protocol selection
src/device/all_reduce.h     ← RING algorithm device code
src/device/primitives.h     ← directRecvReduceDirectSend()
src/device/common_kernel.h  ← reduceCopyPacks() inner loop
```

**Focus on the RING algorithm first** — it's the default and most readable.
Skip NVLS/CollNet/TREE until Stage 5.

**Key questions**:
- What is `ncclInfo`? What fields matter?
- What is a *channel*? How does data get split across channels?
- What are `chunkSteps` and `sliceSteps`?
- What is `ProtoSimple` vs `ProtoLL`?

---

## Stage 3: The Device Kernel System (3–4 days)

**Goal**: Understand how GPU kernels are generated and dispatched.

```
src/device/generate.py      ← read this first, understand the generation logic
src/device/device.h         ← enums that must match generate.py (CRITICAL)
src/device/common_kernel.h  ← BytePack<N>, reduceCopyPacks
src/device/primitives.h     ← the primitive operations
src/device/all_reduce.h     ← how primitives compose into algorithms
```

**Hands-on exercise**: Run:
```bash
make ONLY_FUNCS="AllReduce Sum f32 Ring Simple" VERBOSE=1
```
Watch what `generate.py` emits and how it compiles. Trace one generated `.cu`
file back to its template.

**Key questions**:
- How does `best_kernel()` choose kernel variants?
- What is an *equivalence class* of kernels?
- How does `BytePack<16>` achieve 128-bit memory access?
- Why must enum order in `device.h` match `generate.py`?

---

## Stage 4: Transport Layer (3–4 days)

**Goal**: Understand how data actually moves between GPUs.

Read in this order:

```
src/transport.cc            ← transport selection logic (P2P > SHM > NET)
src/transport/p2p.cc        ← GPU peer-to-peer (the fast path)
src/transport/shm.cc        ← shared memory (intra-node fallback)
src/transport/net_ib.cc     ← InfiniBand RDMA (multi-node)
src/proxy.cc                ← how the CPU proxy thread drives network ops
```

**Key questions**:
- When is P2P chosen vs SHM vs NET?
- What is the listen→connect→accept pattern?
- How do GPU kernels signal the proxy thread to send/recv?
- What is a *proxy operation* (`ncclProxyOp`)?

---

## Stage 5: Topology & Algorithm Selection (2–3 days)

**Goal**: Understand how NCCL picks the right algorithm for the hardware.

```
src/graph/topo.cc           ← hardware graph construction
src/graph/search.cc         ← ring/tree graph search algorithms
src/graph/tuning.cc         ← bandwidth/latency model, algo selection
src/graph/xml.cc            ← topology XML (use NCCL_TOPO_DUMP_FILE to see it)
```

**Hands-on**: Set `NCCL_TOPO_DUMP_FILE=/tmp/topo.xml NCCL_DEBUG=INFO` and run
any example. Read the dumped topology XML and the INFO log to see which
algorithm was chosen and why.

**Key questions**:
- How is the topology graph represented (nodes, links, bandwidth)?
- How does the ring-search algorithm work (`ncclTopoSearchRec`)?
- What determines `TREE` vs `RING` threshold?

---

## Stage 6: Initialization Deep Dive (2 days)

**Goal**: Understand the bootstrap and full init sequence.

```
src/init.cc                 ← ncclCommInitRankFunc, commAlloc, initTransportsRank
src/bootstrap.cc            ← bootstrap ring, bootstrapAllGather
src/group.cc                ← ncclGroupStart/End, deferred launch
```

Full init call chain:

```
ncclCommInitRank()                     init.cc:2189
  ├─ ncclInitEnv()                      — load plugins
  ├─ ncclCudaLibraryInit()              — lazy-load CUDA driver hooks
  └─ ncclCommInitRankDev()             init.cc:2109
       ├─ ncclInit() → bootstrapNetInit()
       ├─ ncclCalloc(&comm)
       └─ ncclAsyncLaunch(ncclCommInitRankFunc)   ← returns immediately

ncclCommInitRankFunc()  [async]        init.cc:1589
  ├─ CUDA setup, ncclInitKernelsForDevice()
  ├─ commAlloc()                        — channels, peer info, memory pools
  ├─ bootstrapInit()                    bootstrap.cc:684
  │    ├─ form bootstrap ring (sockets)
  │    └─ AllGather proxy addresses
  └─ initTransportsRank()              init.cc:879
       ├─ AllGather #1: peer info + version check
       ├─ ncclTopoGetSystem()           — PCI/NVLink/NIC discovery
       ├─ ncclTopoComputePaths()        — bandwidth paths
       ├─ build Ring/Tree/CollNet/NVLS graphs
       ├─ AllGather #3: synchronize nChannels across ranks
       ├─ ncclProxyCreate()            proxy.cc:1869
       │    └─ std::thread(ncclProxyService)   ← proxy thread born here
       └─ ncclTransportRingConnect(), ncclTransportTreeConnect(), ...
```

Now re-read the init sequence with the context of Stages 1–5. Everything
will make much more sense.

---

## Stage 7: Advanced Topics (pick what's relevant)

Once Stages 1–6 are solid, explore based on your interest:

| Topic | Files | When relevant |
|---|---|---|
| NVLS (NVLink Switch) | `src/transport/nvls.cc`, `src/device/all_reduce.h:388` | DGX H100/B200 systems |
| CollNet / SHARP | `src/transport/coll_net.cc` | InfiniBand clusters |
| Symmetric memory | `src/device/symmetric/` | Latest API features |
| GPU-Initiated Net | `src/gin/` | Cutting-edge research |
| Plugin system | `ext-net/`, `ext-tuner/` | Custom hardware/tuning |
| Group ops & fusion | `src/group.cc`, `src/enqueue.cc` | Performance optimization |

---

## Practical Tips

**Use `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL` constantly.** The log output
maps almost 1:1 to the code path — read the log while reading the code.

**Start with 2 GPUs on one node.** Multi-node and NVLink switch paths have
more complexity. Single-node P2P is the simplest transport to reason about.

**Build a mental model of the pipeline:**
```
ncclAllReduce()
  → enqueue → [GPU kernel runs] → proxy drives network → next rank gets data
```
Every complexity in the codebase is either serving this pipeline or optimizing it.

**Follow one data element.** Pick a single float in your send buffer. Ask at
every layer: where is it now, who reads it, who writes it, what synchronizes
access?
