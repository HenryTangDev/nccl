# NCCL Complete Technical Documentation

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Hardware Topology Handling](#2-hardware-topology-handling)
3. [Topology Path Prioritization](#3-topology-path-prioritization)
4. [GPU Direct / P2P Communication](#4-gpu-direct--p2p-communication)
5. [P2P Hardware Details](#5-p2p-hardware-details)

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture

```mermaid
graph TB
    subgraph Client Application
        NCCL_API[ncclAllReduce<br/>ncclBroadcast<br/>ncclAllGather...]
    end

    subgraph NCCL Library Core
        API_Layer[API Layer<br/>src/collectives.cc]
        Enqueue_Layer[Enqueue Layer<br/>src/enqueue.cc]
        Planning_Layer[Planning Layer<br/>src/graph/]
        Execution_Layer[Execution Layer<br/>src/device/]

        API_Layer --> Enqueue_Layer
        Enqueue_Layer --> Planning_Layer
        Planning_Layer --> Execution_Layer
    end

    subgraph Transport Layer
        Transport_API[Transport API<br/>src/transport/transport.cc]
        P2P[GPU P2P<br/>transport/p2p.cc]
        SHM[Shared Memory<br/>transport/shm.cc]
        NET_IB[InfiniBand<br/>transport/net_ib.cc]
        NET_TCP[TCP Socket<br/>transport/net_socket.cc]
        NVLS[NVLink Switch<br/>transport/nvls.cc]
        COLNET[CollNet<br/>transport/coll_net.cc]

        Transport_API --> P2P
        Transport_API --> SHM
        Transport_API --> NET_IB
        Transport_API --> NET_TCP
        Transport_API --> NVLS
        Transport_API --> COLNET
    end

    subgraph Device Kernel Layer
        Kernel_API[Kernel API<br/>src/device/]
        AllReduce_Kernel[AllReduce Kernel<br/>src/device/all_reduce.h]
        AllGather_Kernel[AllGather Kernel<br/>src/device/all_gather.h]
        ReduceScatter_Kernel[ReduceScatter Kernel<br/>src/device/reduce_scatter.h]
        Broadcast_Kernel[Broadcast Kernel<br/>src/device/broadcast.h]
        CodeGen[Code Generator<br/>src/device/generate.py]

        Kernel_API --> AllReduce_Kernel
        Kernel_API --> AllGather_Kernel
        Kernel_API --> ReduceScatter_Kernel
        Kernel_API --> Broadcast_Kernel
        CodeGen --> AllReduce_Kernel
        CodeGen --> AllGather_Kernel
        CodeGen --> ReduceScatter_Kernel
        CodeGen --> Broadcast_Kernel
    end

    Client Application --> API_Layer
    Planning_Layer --> Transport_API
    Execution_Layer --> Kernel_API
    Transport_API --> P2P
    Transport_API --> Proxy_Thread
    Proxy_Thread --> GPU

    P2P --> GPU
    P2P --> PCIE
    SHM --> PCIE

    NET_IB --> IB
    NET_TCP --> TCP

    NVLS --> GPU
    NVLS --> NVLINK

    COLNET --> GPU
    COLNET --> IB
```

### 1.2 Core Components

- **src/collectives.cc** - Public API implementation
- **src/enqueue.cc** - Kernel launch management
- **src/graph/** - Topology detection and algorithm selection
- **src/device/** - Generated GPU kernels
- **src/transport/** - Communication transports

### 1.3 Algorithm Implementations

**AllReduce Algorithms:**
- **RING**: Default, bandwidth-optimal (2*(N-1) steps)
- **TREE**: Latency-optimal (2*log(N) steps)
- **COLLNET_DIRECT**: In-network reduction (IB SHARP)
- **NVLS**: NVLink Switch acceleration
- **NVLS_TREE**: Hybrid NVLS + Tree

---

## 2. Hardware Topology Handling

### 2.1 Topology Detection Process

NCCL automatically discovers hardware topology through a multi-stage process:

```
Stage 1: GPU Detection
┌────────────────────────────────────────────┐
│ Query all CUDA devices                      │
│ Get PCI bus IDs                             │
│ Detect compute capability                   │
│ Query NVLink connections via NVML           │
└────────────────────┬───────────────────────┘
                     │
                     ▼
Stage 2: PCI Topology Analysis
┌────────────────────────────────────────────┐
│ Parse /sys/class/pci* hierarchy            │
│ Discover PCIe switches and bridges         │
│ Build parent-child relationships           │
│ Create PCI nodes in topology graph         │
└────────────────────┬───────────────────────┘
                     │
                     ▼
Stage 3: CPU/NUMA Detection
┌────────────────────────────────────────────┐
│ Identify NUMA domains                      │
│ Detect CPU architecture                    │
│ Query CPU models for bandwidth             │
│ Create CPU nodes for each NUMA domain      │
└────────────────────┬───────────────────────┘
                     │
                     ▼
Stage 4: Network Detection
┌────────────────────────────────────────────┐
│ Discover InfiniBand devices                │
│ Detect TCP interfaces                      │
│ Query PCI bus IDs for NICs                 │
│ Create NIC nodes and connect to PCI tree   │
└────────────────────┬───────────────────────┘
                     │
                     ▼
Stage 5: Path Calculation
┌────────────────────────────────────────────┐
│ Compute optimal paths between all devices  │
│ Calculate bandwidth for each path          │
│ Identify GPU-NIC affinity                  │
└────────────────────────────────────────────┘
```

### 2.2 Topology Graph Structure

```c
struct ncclTopoSystem {
  struct ncclTopoNodeSet nodes[NCCL_TOPO_NODE_TYPES];
};

struct ncclTopoNode {
  int type;  // GPU, PCI, CPU, NIC, NET, NVS
  int64_t id;
  union {
    struct { int dev, rank, cudaCompCap; } gpu;
    struct { int arch, vendor, model; } cpu;
    struct { int dev, port; float bw, latency; } net;
  };
  struct ncclTopoLink links[NCCL_TOPO_MAX_LINKS];
};
```

**Node Types:**
- **GPU**: Graphics Processing Unit
- **PCI**: PCIe switches and bridges
- **CPU**: NUMA domains
- **NIC**: Network Interface Card
- **NET**: Remote network nodes
- **NVS**: NVLink Switch

### 2.3 Path Types and Priority

Path types ordered by preference (fastest to slowest):

```
PATH_LOC (0)  : Local GPU (same)
PATH_NVL (1)  : NVLink direct
PATH_NVB (2)  : NVLink via intermediate GPU
PATH_C2C (3)  : Chip-to-chip
PATH_PIX (4)  : Single PCIe bridge
PATH_PXB (5)  : Multiple PCIe bridges
PATH_P2C (6)  : GPU to NIC via CPU
PATH_PXN (7)  : GPU to NIC via intermediate GPU
PATH_PHB (8)  : PCIe Host Bridge
PATH_SYS (9)  : QPI/UPI between sockets
PATH_NET (10) : Network
PATH_DIS (11) : Disconnected
```

### 2.4 Transport Selection Based on Topology

**Priority Chain:**
```
P2P (GPU Direct) → Shared Memory → Network → Hardware Offload
```

**Transport Decision Logic:**
```c
int p2pLevel = 0;
ncclTopoCheckP2p(comm, topo, rank1, rank2, &p2pLevel);

if (p2pLevel > 0) {
  // P2P is possible, check if NET would be better
  if (p2pLevel >= PATH_NVL || !ncclTopoCheckNet(...)) {
    // Use P2P (faster than NET)
    return TRANSPORT_P2P;
  }
}

// Check shared memory
if (sameHost && topologyAllows) {
  return TRANSPORT_SHM;
}

// Use network transport
return TRANSPORT_NET;
```

### 2.5 Algorithm Selection

**Topology-Aware Algorithm Selection:**

```c
ncclTopoGetAlgoInfo(comm, info, collNetSupport, nvlsSupport, algoInfo);

if (comm->nNodes == 1 && nvlsSupport) {
  // Intra-node, NVLS capable
  algoInfo->algorithm = NCCL_ALGO_NVLS;
} else if (comm->nNodes > 1 && collNetSupport) {
  // Multi-node, CollNet capable
  algoInfo->algorithm = NCCL_ALGO_COLLNET;
} else if (info->nBytes > treeThreshold) {
  // Large message, use ring for bandwidth
  algoInfo->algorithm = NCCL_ALGO_RING;
} else {
  // Small message, use tree for latency
  algoInfo->algorithm = NCCL_ALGO_TREE;
}
```

**Message Size Thresholds:**
- Small messages (<1MB): TREE or NVLS
- Large messages (>1MB): RING
- Hardware offload: Always preferred when available

### 2.6 GPU-NIC Affinity

NCCL automatically matches GPUs to closest NICs:

```c
ncclTopoGetNetDev(system, rank, channelId, &netDev);

// Scores each NIC based on:
// 1. Path type (NVL > PIX > PXB > SYS)
// 2. Path bandwidth
// 3. CollNet support (bonus)
// 4. NUMA locality
```

**Example:**
```
GPU0 (PCIe 01:00.0) → NIC0 (PCIe 01:00.1) [Same switch]
GPU1 (PCIe 02:00.0) → NIC0 (PCIe 01:00.1) [Same switch]
GPU2 (PCIe 81:00.0) → NIC1 (PCIe 81:00.1) [Same switch]
GPU3 (PCIe 82:00.0) → NIC1 (PCIe 81:00.1) [Same switch]
```

---

## 3. Topology Path Prioritization

### 3.1 Path Hierarchy and Numerical Values

NCCL encodes path priorities directly in integer values. **Lower numbers = higher priority (faster connections)**.

```c
// From src/graph/topo.h
#define PATH_LOC   0   // Local (same device)
#define PATH_NVL   1   // NVLink direct
#define PATH_NVB   2   // NVLink via intermediate GPU
#define PATH_C2C   3   // Chip-to-chip (AMD, Intel)
#define PATH_PIX   4   // Single PCIe bridge
#define PATH_PXB   5   // Multiple PCIe bridges
#define PATH_P2C   6   // GPU to NIC via CPU
#define PATH_PXN   7   // GPU to NIC via intermediate GPU
#define PATH_PHB   8   // PCIe Host Bridge
#define PATH_SYS   9   // QPI/UPI between CPU sockets
#define PATH_NET  10   // Network (remote)
#define PATH_DIS  11   // Disconnected
```

**Priority Rule:** NCCL always prefers the path with the **lowest type value** when multiple paths exist between two devices.

### 3.2 How Path Prioritization Works

#### Path Calculation and Scoring

During topology discovery (`ncclTopoComputePaths()`), NCCL computes all possible paths between devices. For each path, it stores:

```c
struct ncclTopoLinkList {
  struct ncclTopoLink* list[NCCL_TOPO_MAX_HOPS];  // Path links
  int count;         // Number of hops
  float bw;          // Bandwidth (GB/s)
  int type;          // Path type (0-11)
};
```

**Path Selection uses type priority then bandwidth:**

```c
// From src/graph/topo.cc:1562-1578
float maxBw = 0;
int minType = PATH_DIS;
for (int i = 0; i < system->nodes[type].count; i++) {
  struct ncclTopoLinkList* path = node->paths[type] + i;
  if (path->bw == 0) continue;  // No path

  // Primary: Maximize bandwidth
  // Secondary: Minimize path type (better connection)
  if (path->bw > maxBw || (path->bw == maxBw && path->type < minType)) {
    maxBw = path->bw;
    minType = path->type;
    bestIndex = i;
  }
}
```

#### Transport Selection Priority

When multiple transport types can connect two GPUs, NCCL uses this priority order:

```
Highest Priority → Lowest Priority

1. P2P (GPU Direct)
   └─ P2P_DIRECT (same process)
   └─ P2P_IPC/CUMEM (cross-process)

2. SHARED MEMORY
   └─ Lock-free ring buffers
   └─ For GPUs on same host

3. NETWORK
   └─ InfiniBand (IB)
   └─ Socket (TCP)

4. HARDWARE OFFLOAD
   └─ NVLS (NVLink Switch)
   └─ CollNet (IB SHARP)
```

### 3.3 GPU-to-GPU Path Priority

When multiple paths connect two GPUs, choose in this order:

```
1. PATH_NVL  (NVLink direct)    ← Best
2. PATH_PIX  (Single PCIe)
3. PATH_PXB  (Multiple PCIe)
4. PATH_PHB  (PCIe Host Bridge)
5. PATH_SYS  (QPI/UPI)
6. PATH_NET  (Network)          ← Worst

Example Decision:
GPU0 → GPU1 has two paths:
- Path A: GPU0 → NVLink → GPU1 (PATH_NVL, 20 GB/s)
- Path B: GPU0 → PCIe → GPU1 (PATH_PIX, 12 GB/s)

NCCL selects: PATH_NVL (lower type value = better)
```

**From `src/graph/paths.cc:329-330`:**
```c
// Default: Don't use P2P across CPU Host Bridges
int p2pLevel = PATH_PXB;  // Allow up to multiple PCIe bridges

// Check if path quality is acceptable
if (path->type <= p2pLevel) {
  *p2p = 1;  // Enable P2P
}
```

### 3.4 Network (NIC) Path Priority

For GPU-to-NIC connections, priority order:

```
1. PATH_NVL  (Direct NVLink to NIC) - Rare
2. PATH_PIX  (Single PCIe bridge)
3. PATH_PXB  (Multiple PCIe)
4. PATH_P2C  (Via CPU - Chip-to-chip)
5. PATH_PXN  (Via another GPU)
6. PATH_PHB  (PCIe Host Bridge)
7. PATH_SYS  (CPU socket cross)
```

**From `src/graph/paths.cc:735-743`:**
```c
int pxnType = ncclParamPxnC2c() ? PATH_P2C : PATH_PXB;

// Enable PXN (P2P via another GPU) if:
if (peerGpu->paths[NET][n].type <= pxnType &&
    peerGpu->paths[GPU][g].type <= PATH_NVL &&  // NVLink!
    ...) {
  // Use this GPU as relay to NIC
}
```

### 3.5 Algorithm Path Priority

**Tree/Ring Construction:**

```
When building rings/trees, NCCL tries paths in order:

For each potential next GPU in ring:
  1. Get path from current GPU to next GPU
  2. If path->type > maxAllowedType: skip
  3. If path->bw < requiredBw: skip
  4. If path creates cycle: skip

Take the valid path with:
  - Highest bandwidth
  - Lowest type (if tie)
```

**From `src/graph/search.cc:181-201`:**
```c
struct ncclGpuScore {
  int interBw;    // Most important
  int interPciBw; // 2nd most
  int interNhops;
  int intraBw;    // 3rd most
  int intraNhops;
  int startIndex; // Least important
};

static int cmpScore(const void* g1, const void* g2) {
  // Compare in order of importance
  int d;
  if ((d = (s2->interBw - s1->interBw))) return d;
  if ((d = (s2->interPciBw - s1->interPciBw))) return d;
  if ((d = (s1->interNhops - s2->interNhops))) return d;
  if ((d = (s2->intraBw - s1->intraBw))) return d;
  if ((d = (s1->intraNhops - s2->intraNhops))) return d;
  return s1->startIndex - s2->startIndex;
}
```

### 3.6 Example Scenarios

#### Scenario 1: DGX A100 System

**Hardware:**
- 8x A100 GPUs
- 6 NVLinks per GPU (NVSwitch topology)
- 2x PCIe switches
- 2x Mellanox NICs

**GPU 0 → GPU 1:**
```
Option A: GPU0 → NVLink → GPU1
  └─ Type: PATH_NVL (1)
  └─ Bandwidth: 20 GB/s
  └─ Selected: YES (lowest type)

Option B: GPU0 → PCIe → GPU1
  └─ Type: PATH_PIX (4) or PATH_PXB (5)
  └─ Bandwidth: 12 GB/s
  └─ Selected: NO (higher type)
```

**GPU 0 → NIC 0:**
```
Option A: GPU0 → PCIe Switch → NIC0
  └─ Type: PATH_PIX (4)
  └─ Bandwidth: 12 GB/s
  └─ Selected: YES

Option B: GPU0 → GPU1 → NIC0 (PXN)
  └─ Type: PATH_PXN (7)
  └─ Bandwidth: 20 GB/s (better!)
  └─ Selected: Depends on bandwidth need vs type penalty
```

#### Scenario 2: Dual-socket PCIe System

**Hardware:**
- 4x GPUs (2 per socket)
- No NVLink
- PCIe switches
- UPI between CPU sockets

**GPU 0 (Socket 0) → GPU 3 (Socket 1):**
```
Path: GPU0 → Socket 0 PCIe → CPU 0
        → UPI → CPU 1
        → PCIe → GPU 3
  └─ Type: PATH_SYS (9)
  └─ Bandwidth: ~6 GB/s (UPI limited)

Result: NCCL prefers GPU0-1, GPU2-3 pairings (same socket)
```

### 3.7 Influencing Path Prioritization

#### Environment Variables

```bash
# Disable transports (forces lower priority)
NCCL_P2P_DISABLE=1    # Skip P2P, use SHM or NET
NCCL_SHM_DISABLE=1    # Skip SHM, use NET
NCCL_NET_DISABLE=1    # Skip NET, use P2P or SHM

# Change P2P threshold (default: PATH_PXB)
NCCL_P2P_LEVEL=PATH_SYS  # Allow cross-socket P2P
# Values: PATH_PIX, PATH_PXB, PATH_PHB, PATH_SYS
```

#### Custom Topology

```bash
# Create custom topology file
NCCL_TOPO_FILE=/path/to/topo.xml

# Manually specify paths and priorities
<system>
  <cpu numaid="0">
    <pci busid="0000:01:00.0">
      <gpu dev="0" rank="0"/>
    </pci>
  </cpu>
</system>
```

### 3.8 Performance Impact

**Bandwidth by Path Type:**
```
Path Type    Bandwidth    Relative Speed
────────────────────────────────────────
PATH_NVL     20 GB/s      3.3x faster
PATH_PIX     12 GB/s      2.0x faster
PATH_PXB     12 GB/s      2.0x faster
PATH_PHB      6 GB/s      1.0x (baseline)
PATH_SYS      6 GB/s      1.0x (baseline)
PATH_NET     12 GB/s      Depends on NIC
```

**AllReduce Performance Impact (8x GPUs, 1GB data):**
```
Best paths (NVLink):      19.5 GB/s
Good paths (PCIe):        11.8 GB/s
Bad paths (via CPU):       6.5 GB/s
Network (no offload):      2-5 GB/s

Difference: 3x between best and worst!
```

### 3.9 Key Takeaways

1. **Path Priority Encoding**: Lower numeric type = better path (0 to 11 scale)
2. **Automatic Selection**: NCCL always chooses best available path by type then bandwidth  
3. **Transport Priority**: P2P → SHM → NET → Hardware Offload, only falls back if unavailable
4. **Configurable**: Can be influenced via env vars, BIOS, device masks, or custom topology
5. **Critical for Performance**: Bad path choices → 3x slower

**Remember:** NCCL's goal is "maximum bandwidth, minimum latency" - path prioritization is fundamental to achieving this!

---

## 4. GPU Direct / P2P Communication

### 3.1 What is GPU Direct / P2P?

**Definition:** Direct GPU-to-GPU memory access without CPU involvement.

**Without P2P:**
```
GPU0 → PCIe → CPU → System RAM → PCIe → GPU1
   └───────┬───────┘      │        └───────┬───────┘
           ↓              ↓                ↓
     High latency    CPU overhead    High latency
     (15-20μs)      (memcpy)        (15-20μs)
     Bandwidth: 6-8 GB/s
```

**With P2P:**
```
GPU0 → PCIe/NVLink → GPU1 (direct)
   └───┬───┘
       ↓
  Low latency (1-2μs)
  High bandwidth (12-40 GB/s)
  Zero CPU involvement
```

### 3.2 Four P2P Modes

NCCL implements four modes to handle various scenarios:

```c
enum p2pType {
  P2P_DIRECT,          // Same process, direct pointers
  P2P_IPC,            // Legacy CUDA IPC (pre-11.3)
  P2P_CUMEM,          // Modern cuMem API (11.3+)
  P2P_INTERMEDIATE    // Via intermediate GPU
};
```

**P2P_DIRECT:**
```c
// Same process, different GPU
cudaDeviceEnablePeerAccess(peerDev, 0);
// Now direct pointer access
```

**P2P_IPC:**
```c
// Different processes
cudaIpcGetMemHandle(&handle, buffer);
// Send handle to other process
cudaIpcOpenMemHandle(&remotePtr, handle, ...);
```

**P2P_CUMEM:**
```c
// Modern cuMem API (CUDA 11.3+)
CUmemGenericAllocationHandle handle;
cuMemCreate(&handle, size, &prop, 0);
cuMemAddressReserve(&ptr, size, 0, 0, 0);
cuMemMap(ptr, size, 0, handle, 0);
// Cross-device mappings via cuMemSetAccess
```

### 3.3 P2P Read vs Write

**P2P Write (Traditional):**
```c
// Sender writes to receiver memory
dst[i] = src[i];
```

**P2P Read (Optimized for NVLink):**
```c
// Receiver reads from sender
dst[i] = src[i];
```

**Selection Logic:**
```c
// Auto-enable read for NVLink-connected Hopper+
if (myInfo->cudaCompCap >= 90 && peerInfo->cudaCompCap >= 90) {
  *useRead = 1;  // P2P Read mode
}

// User override
NCCL_P2P_READ_ENABLE=1  // Force read mode
```

### 3.4 Topology Requirements

**PCIe P2P Requirements:**
- ACS (Access Control Services) disabled in BIOS
- IOMMU disabled or in passthrough mode (AMD)
- GPUs on same PCIe root complex or switch
- Pascal (SM60) or newer architecture

**NVLink P2P:**
- Always available if NVLink connection exists
- No BIOS configuration needed
- Works across PCIe switches

### 3.5 Performance Characteristics

**Latency:**
- Local GPU: ~100 ns
- PCIe P2P: ~1.5 μs
- NVLink P2P: ~0.8 μs
- Multi-hop: ~3-4 μs

**Bandwidth:**
- PCIe Gen3 x16: 12 GB/s
- PCIe Gen4 x16: 24 GB/s
- NVLink 3.0: 20 GB/s per direction
- NVLink 4.0: 40 GB/s per direction

**Example Benchmark (DGX A100):**
```
GPU0 → GPU1 (NVLink):
- 1KB: 1.2 GB/s
- 1MB: 19.5 GB/s
- 100MB: 20.1 GB/s

GPU0 → GPU1 (via system memory):
- 1KB: 0.3 GB/s
- 1MB: 6.2 GB/s
- 100MB: 6.8 GB/s

Improvement: 3x performance
```

### 3.6 MNNVL (Multi-Node NVLink)

**Cross-Node P2P:**
```
GPU0 (Node0) → GPU2 (Node1)
   │                │
   └─NVLink Switch──┘
      Optical cables
      Latency: 5-8μs
      Bandwidth: 15-20 GB/s
```

**How it works:**
- Extends CUDA memory across nodes
- cuMem API manages cross-node mappings
- Appears as single address space
- Transparent to application

---

## 5. P2P Hardware Details

### 3.1 Hardware Component Stack

```
┌────────────────────────────────────────────┐
│ Software Layer                            │
│ ├─ CUDA Driver (Kernel + User Mode)      │
│ └─ CUDA Runtime API                       │
├────────────────────────────────────────────┤
│ GPU Driver (nvidia.ko)                    │
│ ├─ BAR access control                     │
│ ├─ DMA engine management                  │
│ └─ IRQ handling                           │
├────────────────────────────────────────────┤
│ OS Kernel                                  │
│ ├─ IOMMU / VT-d                           │
│ ├─ PCIe ATS                               │
│ └─ DMA coherent mappings                  │
├────────────────────────────────────────────┤
│ PCIe Root Complex                          │
│ ├─ BAR setup                              │
│ ├─ Bus enumeration                        │
│ └─ ACS configuration                      │
├────────────────────────────────────────────┤
│ NVLink Switch (NVSwitch) - Optional       │
│ ├─ Crossbar fabric                        │
│ └─ Full bisection bandwidth               │
├────────────────────────────────────────────┤
│ GPU                                        │
│ ├─ GMMU (GPU MMU)                         │
│ ├─ TLB Caches (L1/L2)                    │
│ ├─ Memory Hub                             │
│ ├─ NVLink/PCIe PHY                        │
│ └─ HBM2/HBM2e DRAM                       │
└────────────────────────────────────────────┘
```

### 3.2 Data Flow During P2P Transfer

**Scenario: GPU0 → GPU1 via NVLink**

```
Step 1: Address Generation
┌────────────────────────────────────────┐
│ GPU0 Thread computes destination:     │
│ dst_addr = 0x7f3b00000000             │
└──────────────┬─────────────────────────┘
               ↓
Step 2: GMMU Translation
┌────────────────────────────────────────┐
│ L1 TLB lookup → MISS                  │
│ L2 TLB lookup → MISS                  │
│ Page table walk (100 cycles)          │
│ Result: GPU1 Physical 0x7f2000        │
└──────────────┬─────────────────────────┘
               ↓
Step 3: Memory Hub Routing
┌────────────────────────────────────────┐
│ Detects remote GPU address            │
│ Routes to NVLink PHY                  │
└──────────────┬─────────────────────────┘
               ↓
Step 4: NVLink Packet Creation
┌────────────────────────────────────────┐
│ Header (20 bytes):                    │
│ ├─ CRC (4B)                           │
│ ├─ Route (2B) → GPU1                 │
│ ├─ Command (1B) → WRITE              │
│ ├─ Address (6B) → 0x7f2000           │
│ └─ Size (2B) → 4 bytes               │
│ Payload (64 bytes):                   │
│ └─ Data + padding                     │
└──────────────┬─────────────────────────┘
               ↓
Step 5: NVLink Transmission
┌────────────────────────────────────────┐
│ 8b/10b encoding                        │
│ Serialization @ 25 GHz                 │
│ On wire: ~3 ns                         │
└──────────────┬─────────────────────────┘
               ↓
Step 6: Reception at GPU1
┌────────────────────────────────────────┐
│ PHY deserializes and decodes           │
│ Validates CRC                          │
│ Routes to local memory hub             │
└──────────────┬─────────────────────────┘
               ↓
Step 7: HBM2 Write
┌────────────────────────────────────────┐
│ GMMU translation (L2 hit)             │
│ DRAM controller: activate row         │
│ Write to sense amplifiers              │
│ Precharge row                          │
│ Total: ~150-200 ns                     │
└──────────────┬─────────────────────────┘
               ↓
         Write Complete!

Total latency: ~250 ns
```

### 3.3 PCIe P2P Flow

```
GPU0 → GPU1 via PCIe Gen3 x16

Step 1: GMMU Translation
└─ Same as NVLink

Step 2: PCI Express TLP Creation
┌────────────────────────────────────────┐
│ TLP Header (16 bytes):                │
│ ├─ Fmt/Type: Posted Write             │
│ ├─ Requester ID: GPU0                 │
│ ├─ Address: GPU1 physical             │
│ └─ Length: 4 bytes                    │
│ Payload: 4-4096 bytes                 │
│ CRC: 4 bytes                          │
└──────────────┬─────────────────────────┘
               ↓
Step 3: PCIe Routing
┌────────────────────────────────────────┐
│ GPU0 → PCIe Switch → Root Complex    │
│ Root Complex → PCIe Switch → GPU1    │
│ (if same root)                        │
└──────────────┬─────────────────────────┘
               ↓
Step 4: Reception
┌────────────────────────────────────────┐
│ GPU1 PCIe controller validates         │
│ Writes to HBM2                          │
│ Same as NVLink from here                │
└────────────────────────────────────────┘

Latency: ~1.5 μs (6x slower than NVLink)
Bandwidth: 12 GB/s (PCIe Gen3)
```

### 3.4 Page Tables and Address Translation

**GPU Page Table Structure:**
```
Virtual Address: 0x7f3a00000000
  ├── PDIndex: Bits 39-47 → Page Directory
  └── PTIndex: Bits 21-38 → Page Table

PDE (Page Directory Entry):
┌──────────────────┐
│ Physical Frame   │
│ PPN: 0x1a34000   │
│ Present: 1       │
└────────┬─────────┘
         │
         ▼
PTE (Page Table Entry):
┌──────────────────┐
│ Physical Frame   │
│ PPN: 0x7f2000    │ ← GPU1 physical address!
│ Present: 1       │
│ Writable: 1      │
│ Remote: 1        │ ← Indicates P2P
└──────────────────┘

Translation Flow:
1. Extract VPN from VA
2. Walk page directory (PDE)
3. Walk page table (PTE)
4. Combine PPN + offset
5. Cached in TLBs

TLB Hierarchy:
┌────────────┐
│ L1 TLB/SM  │→ 128 entries, ~1 cycle
└─────┬──────┘
      ↓
┌────────────┐
│ L2 TLB/GPU │→ 2048 entries, ~15 cycles
└─────┬──────┘
      ↓
┌────────────┐
│ Page Walk  │→ 100-200 cycles
└────────────┘
```

### 3.5 HBM2/HBM2e DRAM Details

**Internal Organization:**
```
GPU Die
  └─ Memory Controllers (8-12)
      └─ HBM2 Stack (8 layers)
          ├─ Layer 0: DRAM
          ├─ Layer 1: DRAM
          ...
          └─ Layer 7: DRAM
              └─ Through-Silicon Via (TSV)

HBM2 Specifications:
├─ 8/16 Gb per die (1-2 GB)
├─ 1024-bit interface per stack
├─ 2.0 Gbps per pin (HBM2)
├─ 3.2 Gbps per pin (HBM2e)
└─ 256-410 GB/s per stack

DRAM Access:
├─ tRCD (RAS to CAS): ~15 ns
├─ tRP (RAS precharge): ~15 ns
├─ tRAS (RAS active): ~35 ns
└─ Total: ~65 ns per random access

Row Buffer:
├─ Each bank has sense amplifiers
├─ If row already open: ~5 ns
├─ If row closed: ~65 ns
└─ NCCL optimizes for sequential access
```

### 3.6 Latency Breakdown

**Complete Timeline (4KB P2P Write):**

```
Component                 Latency    % of Total
───────────────────────────────────────────────
Address generation        5 ns       2%
GMMU translation          50 ns      20%
TLB miss + page walk
Header generation         10 ns      4%
NVLink encode             15 ns      6%
Transmission (3 ns)       3 ns       1%
NVLink decode             15 ns      6%
GMMU at receiver          20 ns      8%
DRAM controller           10 ns      4%
Row activation            45 ns      18%
Data write                50 ns      20%
Precharge                 25 ns      10%
Other overhead            2 ns       1%
───────────────────────────────────────────────
Total                     250 ns     100%
```

**PCIe vs NVLink Comparison:**

```
Metric                 PCIe Gen3    NVLink 3.0    NVLink 4.0
────────────────────────────────────────────────────────────
Latency                1.5 μs       0.8 μs        0.5 μs
Bandwidth              12 GB/s      20 GB/s       40 GB/s
Max hops               No limit     8             8
CPU involvement        Yes          No            No
Protocol overhead      High         Low           Low
PHY power              ~2W          ~10W          ~15W
Cable length           N/A          5m            5m
```

### 3.7 MNNVL (Multi-Node NVLink)

**Cross-Node P2P:**

```
GPU0 (Node0) → GPU2 (Node1) via NVLink fabric

┌────────────────────────────────────────────┐
│ GPU0 Kernel                               │
│ └─ Write to address 0x90000000           │
│ └─ GMMU: Remote Node 1, GPU 2            │
└──────────────┬─────────────────────────────┘
               ↓
┌────────────────────────────────────────────┐
│ External NVLink Switch                    │
│ └─ Optical transceivers                   │
│ └─ Routing tables                         │
│ └─ Forward to Node 1                      │
└──────────────┬─────────────────────────────┘
               ↓
┌────────────────────────────────────────────┐
│ GPU2 (Node1)                              │
│ └─ Receive packet                         │
│ └─ Write to local HBM                     │
└────────────────────────────────────────────┘

Performance:
├─ Latency: 5-8 μs (3x local NVLink)
├─ Bandwidth: 15-20 GB/s (80% of local)
└─ 10x faster than network!
```

---

## 6. Configuration and Performance Tuning

### 3.1 Environment Variables

```bash
# Enable/disable P2P
NCCL_P2P_DISABLE=0        # Enable (default)
NCCL_P2P_DISABLE=1        # Force disable

# P2P read mode (auto-detect based on architecture)
NCCL_P2P_READ_ENABLE=1    # Force read mode
NCCL_P2P_READ_ENABLE=0    # Force write mode

# Disable direct P2P (use shared memory)
NCCL_P2P_DIRECT_DISABLE=1

# Use CUDA memcpy (Copy Engine)
NCCL_P2P_USE_CUDA_MEMCPY=1

# cuMem API settings
NCCL_CUMEM_ENABLE=1
NCCL_CUMEM_HANDLE_TYPE=4  # 4=POSIX_FD, 2=NVLINK
```

### 3.2 Performance Tuning

**BIOS Settings:**
```
1. Disable PCIe ACS (Access Control Services)
   └─ Allows device-to-device DMA
   └─ Critical for P2P functionality

2. IOMMU configuration (AMD)
   └─ Disabled or passthrough mode
   └─ Intel: IOMMU can stay enabled

3. PCIe generation
   └─ Verify x16 width
   └─ Check LinkSta in lspci
```

**NUMA Affinity:**
```bash
# Bind to cores near GPUs
numactl --cpunodebind=0 --membind=0 ./app

# Or use NCCL settings
NCCL_IB_PCI_RELAXED_ORDERING=1
```

**Verification Commands:**
```bash
# Check P2P capability
nvidia-smi topo -p2p r

# Check bandwidth
nvidia-smi topo -p2p w

# Test P2P
/usr/local/cuda/samples/bin/x86_64/linux/release/p2pBandwidthLatencyTest

# Debug NCCL
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=P2P ./app
```

### 3.3 Troubleshooting

**P2P Not Available:**
```bash
# Check ACS is disabled
$ nvidia-smi topo -p2p r
      GPU0    GPU1
GPU0    X     ERR   ← ERR instead of OK

# Check BIOS settings
# lspci -vvv | grep -i acs
```

**Low Bandwidth:**
```bash
# Check PCIe generation
$ lspci -vvv | grep LnkSta
LnkSta: Speed 8GT/s, Width x16

# Should be: Speed 8GT/s (Gen3), Width x16
```

**IPC Handle Failures:**
```bash
# Increase file descriptor limit
ulimit -n 65536

# Check CUDA version compatibility
nvcc --version
```

---

## 7. Performance Characteristics

### 6.1 Latency Comparison

```
Operation                     Latency
────────────────────────────────────
Local GPU memory access       100 ns
P2P via NVLink                800 ns
P2P via PCIe Gen3            1,500 ns
Shared memory               3,000 ns
Network (IB)                2,000 ns
Network (TCP)              15,000 ns
System memory (memcpy)     25,000 ns
```

### 6.2 Bandwidth Comparison

```
Transport                   Bandwidth
────────────────────────────────────
PCIe Gen3 x16                12 GB/s
PCIe Gen4 x16                24 GB/s
PCIe Gen5 x16                48 GB/s
NVLink 3.0 (A100)            20 GB/s
NVLink 4.0 (H100)            40 GB/s
HBM2 memory (A100)        2,000 GB/s
```

### 6.3 Real-World Benchmarks

**DGX A100 (8x A100, NVLink 3.0):**
```
GPU0 → GPU1 P2P:
├─ 1KB: 1.2 GB/s
├─ 1MB: 19.5 GB/s
├─ 100MB: 20.1 GB/s
└─ 1GB: 20.2 GB/s

GPU0 → GPU1 via system memory:
├─ 1KB: 0.3 GB/s
├─ 1MB: 6.2 GB/s
├─ 100MB: 6.8 GB/s
└─ 1GB: 6.9 GB/s

Improvement: 3x (small), 3x (large transfers)
```

**PCIe-Only System:**
```
GPU0 → GPU1 P2P (PCIe Gen3):
├─ 1KB: 0.8 GB/s
├─ 1MB: 11.2 GB/s
└─ 100MB: 11.8 GB/s

GPU0 → GPU1 via system memory:
├─ 1KB: 0.2 GB/s
├─ 1MB: 4.1 GB/s
└─ 100MB: 4.5 GB/s

Improvement: 4x (small), 2.6x (large transfers)
```

---

## 8. Summary

### 7.1 Key Takeaways

**Hardware Topology:**
- ✓ Automatically detected by NCCL
- ✓ Used for optimal algorithm/transport selection
- ✓ GPU-NIC affinity maximizes network performance
- ✓ Bandwidth and latency modeled per link

**P2P Communication:**
- ✓ Fastest intra-node transport
- ✓ 1-2μs latency (vs 15-20μs via CPU)
- ✓ 12-40 GB/s bandwidth (vs 6-8 GB/s via system memory)
- ✓ Four modes handle all scenarios
- ✓ Hardware-driven (no CPU involvement)

**Hardware Components:**
- GPU Cores (SMs) - Execute kernels
- GMMU - Virtual to physical translation
- TLBs - Accelerate address translation
- Memory Hub - Route requests
- NVLink/PCIe PHY - Physical transceivers
- HBM2/e DRAM - Store data

**Performance:**
- Typical intra-node speedup: 2-5x
- Critical for multi-GPU training scaling
- Usually 90-95% of theoretical peak
- Works transparently with NCCL

### 7.2 Files Reference

**Key Implementation Files:**
- `src/graph/topo.cc` - Topology detection
- `src/graph/search.cc` - Ring/tree search
- `src/graph/tuning.cc` - Algorithm selection
- `src/transport/p2p.cc` - P2P implementation
- `src/misc/nvmlwrap.cc` - NVLink detection
- `src/include/p2p.h` - P2P API definitions

---

## Appendix: Quick Reference

### A.1 Debug Commands

```bash
# Topology check
nvidia-smi topo -m

# P2P capabilities
nvidia-smi topo -p2p r

# P2P bandwidth
nvidia-smi topo -p2p w

# NCCL debug
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=P2P,GRAPH ./app

# PCIe info
lspci -vvv | grep -A 10 "VGA compatible"
```

### A.2 Environment Variables

```bash
NCCL_P2P_DISABLE=0              # Enable P2P
NCCL_SHM_DISABLE=0              # Enable SHM
NCCL_IB_DISABLE=0               # Enable InfiniBand
NCCL_NET_GDR_LEVEL=5            # GPU Direct RDMA
NCCL_TREE_THRESHOLD=1M          # Tree algorithm threshold
NCCL_DEBUG=INFO                 # Enable debug output
```

### A.3 Common Performance Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| P2P shows ERR | ACS enabled | Disable ACS in BIOS |
| Low P2P BW | PCIe at x8 | Check slot/cable |
| High latency | Wrong algorithm | Check NCCL_ALGO |
| NCCL hangs | Topology issue | Run NCCL_DEBUG=INFO |

---

*Document version: 1.0*
*NCCL Version: 2.29.3*
*Last updated: 2026-02-04*
