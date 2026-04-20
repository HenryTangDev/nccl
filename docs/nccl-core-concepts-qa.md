# NCCL Core Concepts — Q&A

Grounded in source code from `src/include/comm.h`, `src/include/info.h`, `src/include/device.h`, `src/include/collectives.h`, `src/device/primitives.h`, `src/device/prims_ll.h`, `src/device/prims_simple.h`, and `src/enqueue.cc`.

---

## What is a Rank?

The simplest of the three. A rank is just an **integer index** in `[0, nRanks)` that identifies one GPU within a communicator.

```c
// comm.h:546-547
int rank;    // my rank in the communicator
int nRanks;  // number of GPUs in communicator
```

There is no `ncclRank` struct — just an `int`. Every `ncclComm` knows its own `rank` and the total `nRanks`. There is also a secondary `localRank` / `localRanks` for within-node identity (relevant for transport selection), but the primary identity is the global rank.

---

## What is a Communicator?

A communicator is the **central object** representing a named group of GPUs that will collectively communicate. It is one `ncclComm` struct per GPU — 504+ fields spanning topology, transport, channels, planner, proxy, and tuner.

Key fields from `comm.h:504+`:

```c
struct ncclComm {
  int rank;                              // this GPU's index
  int nRanks;                            // total GPUs in the group
  int cudaDev;                           // which CUDA device
  struct ncclChannel channels[MAXCHANNELS]; // the actual pipelines
  struct ncclTopoSystem* topo;           // detected hardware graph
  struct ncclPeerInfo* peerInfo;         // capability info for every other rank
  struct ncclKernelPlanner planner;      // task queue between GroupStart/End
  struct ncclProxyState* proxyState;     // background CPU thread for networking
  struct ncclTopoGraph graphs[NCCL_NUM_ALGORITHMS]; // one path graph per algo
  int nChannels;                         // how many channels are connected
  int collChannels;                      // how many channels used for collectives
  float bandwidths[...][...][...];       // per-algo/proto bandwidth table
  ncclNet_t* ncclNet;                    // plugged-in network transport
};
```

The communicator is one-per-GPU: 8 GPUs → 8 `ncclComm` objects, all linked into the same logical group via a shared `commHash`. They don't share a single host object.

---

## What is a Channel?

A channel is an **independent, complete communication pipeline** through all ranks — with its own ring order, tree shape, and per-peer send/recv buffers. Think of it as one "lane" of data movement.

```c
// comm.h:139-161
struct ncclChannel {
  struct ncclRing ring;            // for RING algo: prev/next/userRanks ordering
  struct ncclTree tree;            // for TREE algo: up + down[3] connections
  struct ncclDirect collnetDirect; // for COLLNET_DIRECT
  struct ncclTree collnetChain;    // for COLLNET_CHAIN
  struct ncclNvls nvls;            // for NVLS

  struct ncclChannelPeer** peers;  // per-rank connection handles (send+recv buffers, transport)
  int id;                          // index of this channel
  uint32_t workFifoProduced;       // position in GPU work FIFO
};
```

The `ring` sub-struct defines this channel's specific rank ordering:

```c
// device.h:169-181
struct ncclRing {
  int prev;        // rank to my left in this ring
  int next;        // rank to my right in this ring
  int* userRanks;  // full ring order [nRanks], starting from me
  int index;       // my position in this ring
};
```

Each channel has its own ring permutation. Different channels traverse ranks in different orders — NCCL's topology search deliberately staggers them to exploit different physical links and maximize aggregate bandwidth.

**How many channels?** Up to `MAXCHANNELS = 64` (device.h:86). The actual count is:
```c
int nChannels;     // number of connected channels (hardware-limited)
int collChannels;  // number used for collectives (tuner-controlled)
```

### Communicator / Rank / Channel relationship

```
ncclComm (one per GPU)
├── rank = 3, nRanks = 8
├── channels[0]      ← ring order [3,4,5,6,7,0,1,2], own buffers
├── channels[1]      ← ring order [3,2,1,0,7,6,5,4], own buffers
└── channels[2..N-1] ← more channels for more parallelism
```

Multiple channels allow pipelining and parallelism: a large message is split into N chunks, one per channel, all running concurrently in the same kernel launch.

---

## How does data get split across channels?

A channel owns a **contiguous slice of the element array**. `src/enqueue.cc` computes per-channel element counts as `countLo`, `countMid`, `countHi`, then each channel's proxy op gets a `loopOffset` and `channelSize` marking its window:

```c
// enqueue.cc ~737
proxyOp->loopOffset  = (countLo + (c - channelLo - 1) * countMid) * elementSize;
proxyOp->channelSize = countMid * elementSize;
```

For a 512 MB AllReduce over 4 channels:

```
total data:  [0 ─────────────────── 512 MB]

channel[0]:  [0──────128MB]
channel[1]:          [128──256MB]
channel[2]:                  [256──384MB]
channel[3]:                          [384──512MB]
```

Each channel independently runs its own ring pipeline over its slice. They all execute in the same kernel launch — CUDA blocks for channel `c` read `loopOffset[c]` from shared memory and operate on that subarray.

---

## What is `ncclInfo`?

`src/include/info.h` defines `ncclInfo` as a **short-lived parameter-passing struct** — stack-allocated by each public API function and consumed by `ncclEnqueueCheck`. It is never stored.

```c
struct ncclInfo {
  ncclFunc_t    coll;        // which collective (AllReduce, AllGather, etc.)
  const char*   opName;
  const void*   sendbuff;
  void*         recvbuff;
  size_t        count;       // number of elements
  ncclDataType_t datatype;
  ncclRedOp_t   op;
  int           root;
  ncclComm_t    comm;
  cudaStream_t  stream;
  // ← the two fields that control pipelining depth:
  int           chunkSteps;
  int           sliceSteps;
};
```

Fields that drive decisions downstream: `coll`, `count`, `comm`, `chunkSteps`, `sliceSteps`. Everything else is passed through to the kernel.

---

## What are `chunkSteps` and `sliceSteps`?

These only activate for **SIMPLE protocol + RING algorithm** (`enqueue.cc:2131`). All other algo/proto combinations force both to `1`.

Constants from `src/include/collectives.h`:

```c
#define NCCL_STEPS            8    // ring buffer depth (number of slots)
#define ALLREDUCE_CHUNKSTEPS  4    // NCCL_STEPS / 2
#define ALLREDUCE_SLICESTEPS  2    // NCCL_STEPS / 4
```

They define a **two-level pipeline** within one channel's data window:

```c
// enqueue.cc:2130-2133
stepSize  = buffSizes[protocol] / NCCL_STEPS;  // bytes per one ring-buffer slot
chunkSize = stepSize * chunkSteps;             // total bytes per rank-to-rank hop
sliceSize = stepSize * sliceSteps;             // unit sent per pipeline tick
```

Hierarchy:

```
channel window (channelSize bytes)
 └─ loop iterations  (advance by nRanks × chunkSize each)
      └─ chunk  = chunkSteps × stepSize   (one rank's worth per ring pass)
           └─ slice = sliceSteps × stepSize  (one pipelined send unit)

For AllReduce, NCCL_STEPS=8, chunkSteps=4, sliceSteps=2:
  chunk  = 4 slots
  slice  = 2 slots
  slices per chunk = chunkSteps / sliceSteps = 2
```

**Why two levels?** While slice N is being sent over the network, slice N+1 is being loaded by the GPU. This overlap hides latency. More `chunkSteps` = larger in-flight window = more pipelining, at the cost of more buffer usage.

---

## What is `ProtoSimple` vs `ProtoLL`?

They differ in **how the receiver knows data is ready**. Defined in `src/device/primitives.h`.

### ProtoSimple — separate data buffer + counter

```
sender buffer layout (stepSize bytes per slot, NCCL_STEPS slots):
┌──────────────────┬──────────────────┬─────┐
│   slot 0 (data)  │   slot 1 (data)  │ ... │   ← pure data, no flags mixed in
└──────────────────┴──────────────────┴─────┘

tail counter ← sender writes it *after* data is fully written
               receiver polls counter, then reads data
```

- **Full bandwidth**: 100% of buffer bytes are payload data
- **Sync cost**: receiver must poll a separate `tail` pointer — an extra cache line touch before data access
- **Direct support**: can bypass the buffer entirely and write straight into the remote GPU's memory (NVLink DirectWrite/DirectRead)
- **Best for**: large messages where bandwidth dominates

### ProtoLL — data and flag interleaved in 8-byte lines

The `ncclLLFifoLine` union (`src/include/device.h:70-88`):

```c
union ncclLLFifoLine {
  struct {
    uint32_t data1;   // 4 bytes of payload
    uint32_t flag1;   // 4-byte validity flag  ← same 8-byte atomic word as data1
    uint32_t data2;   // 4 bytes of payload
    uint32_t flag2;   // 4-byte validity flag
  };
};  // 16 bytes total; only 8 bytes are data (50% efficiency)
```

The flag and its data occupy the same 8-byte naturally-aligned word. On RDMA hardware (IB), an 8-byte store is atomic — so when the receiver sees `flag == expected_step`, the adjacent 4 bytes of data are guaranteed to be present too. **No separate synchronization step needed**.

```c
// prims_ll.h — receiver loop:
while (readLL(recvPtr(i), flag) != recvFlag(i))
  { /* spin on flag — data is in the SAME 16-byte line, one L1 miss total */ }
```

- **50% bandwidth** — half the buffer is flags
- **Lower latency**: no separate head/tail pointer; receiver spins directly on data cache lines
- **No direct mode**: inline flags rule out bypassing the buffer
- **Best for**: small messages where round-trip latency dominates

### ProtoLL128 — 128-byte cacheline with flag at the end

Uses a full 128-byte NVLink cacheline: 120 bytes of data + 8 bytes of flag at the end (~93% efficiency). The receiver waits for the flag word at offset 120 to confirm the entire 128-byte line is valid. Better throughput than LL for medium-sized messages on NVLink hardware.

### Comparison table

| | `ProtoSimple` | `ProtoLL` | `ProtoLL128` |
|---|---|---|---|
| Data efficiency | 100% | 50% (4B data + 4B flag per 8B) | ~93% (120B data + 8B flag per 128B) |
| Sync mechanism | Poll `tail` counter (separate cacheline) | Flag embedded in data word | Flag at end of 128B NVLink line |
| `calcBytePerStep()` | `buffSizes[SIMPLE] / NCCL_STEPS` | `buffSizes[LL] / NCCL_STEPS / 2` | scaled by `DATAELEMS/LINEELEMS` |
| Direct GPU writes | Yes (NVLink bypass) | No | No |
| Use case | Large messages | Small messages | Medium messages (NVLink) |

### How protocol is selected

```c
// enqueue.cc:2130-2135
stepSize  = comm->buffSizes[info->protocol] / NCCL_STEPS;
chunkSize = stepSize * chunkSteps;
if (info->protocol == NCCL_PROTO_LL)    chunkSize /= 2;           // half is flags
if (info->protocol == NCCL_PROTO_LL128) chunkSize = (chunkSize / NCCL_LL128_LINEELEMS)
                                                   * NCCL_LL128_DATAELEMS;
```

Protocol thresholds are tunable via `NCCL_LL_THRESHOLD` and `NCCL_LL128_THRESHOLD` environment variables.
