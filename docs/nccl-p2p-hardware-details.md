# P2P Hardware Data Flow and Components - Detailed Analysis

## Overview

This document explains the **hardware-level details** of GPU Direct P2P transfers in NCCL, including all components involved, data flow paths, and low-level mechanisms. We'll trace data from source GPU memory to destination GPU memory through the actual hardware.

---

## 1. Hardware Components in P2P Communication

### Primary Components Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                    P2P Hardware Stack                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                     Software Layer                           │ │
│  │  ┌────────────────────────────────────────────────────────┐  │ │
│  │  │  CUDA Driver (Kernel Mode + User Mode)                │  │ │
│  │  │  - CUDA API (cudaMemcpyAsync, cudaMemset, etc.)      │  │ │
│  │  │  - Virtual memory management                          │  │ │
│  │  └────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────┬─────────────────────────────────────┘ │
│                           │                                       │
│                           └─ CUDA Driver API (libcuda.so)         │
│                           └─ CUDA Runtime API (libcudart.so)      │
└───────────────────────────┬───────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
┌─────────────▼─────────────┐  ┌──────────▼──────────┐
│    GPU Driver             │  │  nvidia-peermem     │
│  (nvidia.ko)              │  │  (Legacy: nvidia-uvm)│
│                           │  │                       │
│ - BAR access control      │  │ - P2P mappings       │
│ - DMA engine mgmt         │  │ - Page table updates │
│ - IRQ handling            │  │ - Memory registration│
│ - Power management        │  └─────────┬────────────┘
└─────────────┬─────────────┘            │
              │                          │
┌─────────────▼──────────────────────────▼─────────────────────┐
│                                                              │
│              Operating System Kernel                         │
│                (Linux/Windows/etc.)                          │
│                                                              │
│  - IOMMU / VT-d configuration                               │
│  - PCIe ATS (Address Translation Services)                  │
│  - sysfs PCI interface (/sys/bus/pci/)                      │
│  - DMA coherent mappings                                    │
│                                                              │
└──────────────┬───────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────┐
│                                              │
│          PCIe Root Complex                   │
│   (Intel PCH, AMD chipset, etc.)            │
│                                              │
│  - PCIe configuration space                 │
│  - BAR (Base Address Register) setup        │
│  - Bus enumeration                          │
│  - ACS (Access Control Services)            │
│                                              │
└──────────────┬───────────────────────────────┘
               │ PCIe Fabric (Wires or NVLink)
               │
┌──────────────▼──────────────────────────────────────┐
│                                                      │
│            ╔═══════════════════════╗               │
│            ║   NVLink Switch       ║               │
│            ║   (NVSwitch)          ║   (Optional)  │
│            ║   - Full crossbar     ║               │
│            ║   - Up to 18 ports    ║               │
│            ║   - 900 GB/s bisec.   ║               │
│            ╚═══════════════════════╝               │
│                                                      │
└──────────────┬────────────────────────┬──────────────┘
               │                        │
┌──────────────▼────────────┐  ┌────────▼────────┐
│                            │  │                 │
│      GPU 0 (Source)       │  │   GPU 1 (Dest)  │
│                            │  │                 │
│  ┌─────────────────────┐  │  │  ┌────────────┐ │
│  │ GPU Memory (VRAM)   │◄──────────────────┐   │
│  │ - FB Memory         │  │  │  │ GPU Core   │ │
│  │ - Memory Controller │  │  │  └────────────┘ │
│  └─────────┬───────────┘  │  │        ▲        │
│            │              │  │        │ L1/L2    │
│            │ PCIe/NVLink  │  │        │ Cache    │
│  ┌─────────▼───────────┐  │  │  ┌─────▼────┐    │
│  │   GPU Memory Hub    │  │  │  │  GMMU    │    │
│  │   - MMU             │  │  │  │  (GPU MMU)│   │
│  │   - NVLink PHY      │  │  │  └──────────┘    │
│  │   - PCIe PHY        │  │  │        │         │
│  └─────────────────────┘  │  │        │         │
│                            │  │        │         │
└────────────────────────────┘  └────────┼─────────┘
                                          │
                                    ┌─────▼──────┐
                                    │            │
                                    │  PCIe Slot │
                                    │  Connector │
                                    │            │
                                    └────────────┘
```

### Component Details

#### 1. GPU Memory (VRAM)
- **HBM2/HBM2e/HBM3**: High Bandwidth Memory
- **Capacity**: 16GB (A100), 40-80GB (A100), 80-94GB (H100)
- **Bandwidth**: Up to 3 TB/s (H100), 2 TB/s (A100)
- **FB (Frame Buffer)**: Physical GPU memory
- **ECC Protection**: Automatic error correction

#### 2. GPU Memory Controller
- **HBM Controllers**: 8-12 memory controllers per GPU
- **Memory Channels**: 4096-8192-bit wide interface
- **Row/Column Addressing**: DDR-style burst transfers
- **DRAM Protocol**: DDR5-based HBM protocol

#### 3. GMMU (GPU Memory Management Unit)
- **Virtual Addressing**: 49-bit virtual address space
- **Page Tables**: Similar to CPU MMU
- **Page Size**: 4KB, 64KB, 2MB
- **TLB Hierarchy**: L1 and L2 TLBs
- **ATS**: Address Translation Services
- **Address Translation**: Virtual → Physical (GPU memory)

#### 4. GPU Memory Hub
- **Central Interconnect**: Connects all GPU components
- **NVLink PHY**: Physical NVLink transceivers (up to 18)
- **PCIe PHY**: PCIe physical layer (up to 32 lanes)
- **Bandwidth Arbitration**: Manages concurrent accesses
- **Coherence**: Maintains cache coherency across GPUs

#### 5. PCIe PHY and Controller
- **PCIe Generation**: Gen3 (8 GT/s), Gen4 (16 GT/s), Gen5 (32 GT/s)
- **Lane Count**: x1, x4, x8, x16
- **Root Complex**: Connection to CPU/chipset
- **ATS**: Address Translation Services
- **ACS**: Access Control Services (must be disabled for P2P)

#### 6. NVLink PHY
- **NVLink Generations**:
  - **NVLink 1**: 24 GB/s (P100)
  - **NVLink 2**: 50 GB/s (V100)
  - **NVLink 3**: 50 GB/s (A100, improved efficiency)
  - **NVLink 4**: 90 GB/s (H100 with NVLink4)
  - **NVLink 5**: 200 GB/s (H200/B100)
- **Link Count**: 4-18 links per GPU
- **Bidirectional**: Full-duplex communication
- **Logical Layer**: Handles packetization

#### 7. NVLink Switch (NVSwitch)
- **2nd Gen NVSwitch**: 2.4 TB/s switching capacity
- **3rd Gen NVSwitch**: 4.8 TB/s switching capacity
- **Ports**: 18 ports (2nd gen), 64 ports (3rd gen)
- **Crossbar**: Full bandwidth between any ports
- **Topology Support**: Fat tree, mesh, torus

---

## 2. Data Flow During P2P Transfer

### Example: CPU Initiates GPU 0 → GPU 1 Transfer

#### Phase 1: Setup (One-time, during ncclCommInitRank)

```
Step 1: Memory Allocation
────────────────────────────────────────
CPU (Process 0):
  cudaMalloc(&buffer_gpu0, 1GB);

Result:
  GPU 0 VRAM: [████████████░░░░░░░░░░░░░░] 1GB allocated
Address: 0x7f3a00000000

Step 2: Enable Peer Access
────────────────────────────────────────
CPU (Process 0):
  cudaDeviceEnablePeerAccess(gpu1, 0);

Hardware Actions:
1. CPU writes to GPU 0 MMIO
2. GPU 0 updates GMMU page tables
3. GPU 0 sends PCIe P2P enable message to GPU 1
4. GPU 1 acknowledges and updates own page tables
5. Both GPUs add each other to peer list

Result:
  GPU 0 can now access GPU 1 memory space
  GPU 1 can now access GPU 0 memory space

Step 3: Create IPC Handle (Cross-process)
────────────────────────────────────────
CPU (Process 0):
  cudaIpcGetMemHandle(&handle, buffer_gpu0);

Hardware Actions:
1. GPU MMU creates exportable handle
2. Contains physical address + GPU ID
3. Compressed into 128-bit handle
4. Ready for sharing with other processes

Result:
  handle = {0x12345678abcdef00...} (128 bytes)
```

#### Phase 2: Data Transfer (ncclAllReduce, etc.)

```
Scenario: Transfer 1.0 GB from GPU 0 to GPU 1

Step 1: CPU Launches CUDA Kernel on GPU 0
────────────────────────────────────────
CPU (Process 0):
  myKernel<<<blocks, threads, 0, stream>>>(src, dst);

Hardware Path:
1. CPU writes kernel launch parameters to GPU 0 command buffer
   └─ Command Buffer Address: GPU 0 MMIO region
   └─ Payload: Grid dims, block dims, kernel function pointer

2. GPU 0 Command Processor fetches commands

Step 2: Kernel Executes on GPU 0
────────────────────────────────────────
GPU 0 (SMs - Streaming Multiprocessors):
  Each thread performs:
    dst[i] = src[i] * 2;  // (example operation)

But dst is in GPU 1 memory!

Hardware Actions per Thread:

┌────────────────────────────────────────────────────────────────┐
│ GPU 0 Thread (e.g., Thread 0 in Warp 0)                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 1. Virtual Address Generation                                 │
│    Thread computes destination address:                      │
│      dst_addr = 0x7f3b00000000 + thread_id * 4               │
│                                                                 │
│ 2. GMMU Address Translation                                   │
│    Thread accesses GMMU (GPU Memory Management Unit)         │
│                                                                 │
│    GMMU L1 TLB Lookup:                                       │
│      └─ Key: VPN (Virtual Page Number) = 0x7f3b000           │
│      └─ Result: TLB MISS (first access)                      │
│                                                                 │
│    GMMU L2 TLB Lookup:                                       │
│      └─ Key: VPN = 0x7f3b000                                 │
│      └─ Result: TLB HIT!                                     │
│      └─ PPN (Physical Page Number) = GPU 1:0x1a2000          │
│                                                                 │
│    [GMMU caches this translation in L1 TLB]                    │
│                                                                 │
│ 3. Memory Request Generation                                   │
│    GPU 0 generates memory write packet:                        │
│      Command: WRITE                                          │
│      Address: GPU 1 Physical = 0x1a2000 + offset             │
│      Size: 4 bytes                                           │
│      Data: source_value * 2                                  │
│      Source: SM 0, Warp 0, Thread 0                          │
│                                                                 │
│ 4. Request Routing                                             │
│    Memory Hub receives request and looks at physical address │
│                                                                 │
│    IF address is in GPU 0 local memory:                      │
│      └─ Route to local HBM                                   │
│      └─ Write completes in ~100ns                            │
│    ELSE IF address is in GPU 1 memory:                       │
│      └─ Route to NVLink / PCIe                               │
│      └─ Continue to Step 5                                   │
│                                                                 │
│ 5. NVLink Packet Generation (if NVLink)                       │
│    GPU 0 converts memory write to NVLink packet:             │
│                                                                 │
│    NVLink Header:                                              │
│      ┌─────────────────────────────────────────────────────┐   │
│      │ CRC (4 bytes) - Error detection                      │   │
│      ├─────────────────────────────────────────────────────┤   │
│      │ Route (2 bytes) - Destination GPU ID                 │   │
│      ├─────────────────────────────────────────────────────┤   │
│      │ Command (1 byte) - WRITE/READ/ATOMIC                │   │
│      ├─────────────────────────────────────────────────────┤   │
│      │ Tag (1 byte) - Transaction ID                       │   │
│      ├─────────────────────────────────────────────────────┤   │
│      │ Address (6 bytes) - Physical address in dest GPU   │   │
│      ├─────────────────────────────────────────────────────┤   │
│      │ Size (2 bytes) - Payload size in bytes             │   │
│      └─────────────────────────────────────────────────────┘   │
│                                                                 │
│    NVLink Payload:                                             │
│      ┌─────────────────────────────────────────────────────┐   │
│      │ Data (4 bytes) - The value to write                 │   │
│      │ Pad (60 bytes) - NVLink is 64-byte granularity      │   │
│      └─────────────────────────────────────────────────────┘   │
│                                                                 │
│    Total packet size: 80 bytes (header + payload)            │
│                                                                 │
│ 6. NVLink PHY Transmission                                     │
│    GPU 0's NVLink PHY encodes and serializes the packet:     │
│                                                                 │
│    Encoding Steps:                                             │
│      └─ 8b/10b encoding (adds 2 bits per byte for DC balance)│
│      └─ Serialization (converts parallel to serial)           │
│      └─ Transmitter amplifies signal                         │
│                                                                 │
│    NVLink 3.0 Parameters:                                      │
│      └─ Data Rate: 50 Gbps per direction (25 GB/s per link)  │
│      └─ SerDes: 25 GHz clock                                  │
│      └─ Encoding: 8b/10b                                      │
│      └─ Effective Data: 40 Gbps (5 GB/s per direction)       │
│                                                                 │
│    NVLink Switch (if not direct):                              │
│      └─ Receives packet on Port X                             │
│      └─ Looks up destination in routing table                 │
│      └─ Forwards to Port Y connected to GPU 1                 │
│      └─ Store-and-forward latency: ~50-100ns                  │
└────────────────────────────────────────────────────────────────┘

Step 3: NVLink Reception at GPU 1
────────────────────────────────────────

┌───────────────────────────────────────────────────────────────┐
│ GPU 1 NVLink PHY                                              │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│ 1. Signal Reception                                            │
│    └─ Differential receiver captures serial signal           │
│    └─ Clock recovery extracts timing from data               │
│    └─ Deserializes to parallel data                          │
│                                                                │
│ 2. 8b/10b Decoding                                             │
│    └─ Converts 10-bit symbols back to 8-bit data             │
│    └─ Detects and corrects single-bit errors                 │
│    └─ Checks CRC for multi-bit errors                        │
│                                                                │
│ 3. Packet Decoding                                             │
│    └─ Extracts header information                            │
│    └─ Validates destination GPU ID = GPU 1                   │
│    └─ Validates command = WRITE                              │
│    └─ Extracts destination address                           │
│                                                                │
│ 4. Memory Hub Routing                                          │
│    └─ Memory Hub determines this is local memory access      │
│    └─ Routes to local HBM controllers                        │
│                                                                │
│ 5. GMMU Check (Again)                                          │
│    └─ GPU 1 checks if address is in local memory             │
│    └─ Address passes GMMU (already mapped)                   │
│    └─ Gets physical address in local HBM                     │
└───────────────────────────────────────────────────────────────┘

Step 4: HBM2 Write in GPU 1
────────────────────────────────────────

┌────────────────────────────────────────────────────────────────┐
│ HBM2 Memory Controller (GPU 1)                                │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 1. Receive Write Command                                        │
│    └─ Address: Bank 3, Row 0x1a20, Column 0x0                │
│    └─ Data: computed_value                                    │
│    └─ Size: 4 bytes                                           │
│                                                                 │
│ 2. HBM2 Protocol                                                │
│    HBM2 (High Bandwidth Memory 2) protocol:                    │
│                                                                 │
│    Timing Diagram:                                              │
│      ACT (Activate) → READ/WRITE → PRE (Precharge)            │
│      └─ Opens a row                                          │
│      └─ Performs read/write                                  │
│      └─ Closes row                                           │
│                                                                 │
│    HBM2 Characteristics:                                       │
│      └─ 8 independent channels                                │
│      └─ 128-bit wide per channel                              │
│      └─ DDR (Double Data Rate)                               │
│      └─ 1.6-2.0 Gbps per pin                                  │
│      └─ ~460 GB/s total bandwidth (A100)                      │
│                                                                 │
│ 3. Write Operation                                              │
│    └─ Activate row 0x1a20 in bank 3                          │
│    └─ Wait tRCD (Row to Column Delay): ~15ns                 │
│    └─ Write data to column 0x0                               │
│    └─ Wait tWR (Write Recovery): ~10ns                       │
│    └─ Precharge row                                          │
│    └─ Wait tRP (RAS Precharge): ~10ns                        │
│                                                                 │
│ 4. Data Storage                                                 │
│    └─ Data is stored in DRAM cells (capacitors)              │
│    └─ Sense amplifiers detect charge levels                  │
│    └─ Data is now persistent in GPU 1 memory                 │
│       Address: 0x1a2000                                       │
│       Value: computed_value                                 │
└────────────────────────────────────────────────────────────────┘

Step 5: Write Acknowledgment
────────────────────────────────────────

For Fire-and-Forget Writes (CUDA memory model):
┌────────────────────────────────────────┐
│ No acknowledgment needed              │
│                                        │
│ Writes are posted and acknowledged     │
│ at PHY/link layer (CRC check)         │
│                                        │
│ GPU 0 continues execution             │
└────────────────────────────────────────┘

For Reads or Atomic Operations:
┌────────────────────────────────────────┐
│ ACK Packet sent back via NVLink       │
│                                        │
│ GPU 1 → NVLink Switch → GPU 0        │
│                                        │
│ Contains:                             │
│  - Tag (matches original request)     │
│  - Data (for reads)                   │
│  - Status (success/error)             │
└────────────────────────────────────────┘

Repeat for All Threads
────────────────────────────────────────

GPU 0 executes thousands of threads simultaneously:

Warp 0 Thread 0: Writes to GPU 1:0x1a2000
Warp 0 Thread 1: Writes to GPU 1:0x1a2004
Warp 0 Thread 2: Writes to GPU 1:0x1a2008
Warp 0 Thread 3: Writes to GPU 1:0x1a200c
... (32 threads per warp)

Warp 1 Thread 0: Writes to GPU 1:0x1a2080
Warp 1 Thread 1: Writes to GPU 1:0x1a2084
... (1000s of warps)

All executing in parallel across GPU 0's SMs!
```

---

## 3. PCIe-based P2P (Alternative to NVLink)

```
When GPUs are not NVLink-connected (e.g., standard PCIe system)

Hardware Components:
┌──────────────────────────────────────────────────────────────┐
│                    PCIe Topology                             │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  CPU Socket 0                                               │
│    └─ PCIe Root Complex 0                                  │
│        ├─ PCIe Switch 0                                     │
│        │   ├─ GPU 0 (x16 slot)                             │
│        │   └─ GPU 1 (x16 slot)                             │
│        │                                                    │
│        └─ PCIe Switch 1                                     │
│            ├─ NIC                                           │
│            └─ Other devices                                 │
│                                                               │
│  CPU Socket 1 (if dual-socket)                              │
│    └─ PCIe Root Complex 1                                  │
│        └─ PCIe Switch 2                                     │
│            └─ GPU 2 (x16 slot)                             │
│                                                               │
└──────────────────────────────────────────────────────────────┘

Data Transfer: GPU 0 → GPU 1 via PCIe
────────────────────────────────────────

1. GPU 0 issues memory write to GPU 1 address
   └─ Address: 0x7f3b00000000 (GPU 1 physical)

2. PCIe Transaction Layer Packet (TLP) Creation:
   └─ TLP Header (16 bytes):
       ┌─────────────────────────────────────┐
       │ Fmt/Type (1 byte) - Posted Write   │
       ├─────────────────────────────────────┤
       │ TC/Attr (1 byte) - Traffic Class   │
       ├─────────────────────────────────────┤
       │ Length (2 bytes) - Payload size    │
       ├─────────────────────────────────────┤
       │ Requester ID (2 bytes) - GPU 0 ID  │
       ├─────────────────────────────────────┤
       │ Tag (1 byte) - Transaction ID      │
       ├─────────────────────────────────────┤
       │ Address (6 bytes) - Dest in GPU 1  │
       ├─────────────────────────────────────┤
       │ Last DW BE/1st DW BE (2 bytes)     │
       └─────────────────────────────────────┘
   └─ TLP Data (4-4096 bytes):
       └─ Actual data to write
       └─ CRC (4 bytes) - End-to-end CRC

3. PCIe Routing:
   GPU 0 → PCIe Switch 0 → PCIe Root Complex 0
     └─ Switch forwards based on routing tables
     └─ Root Complex detect destination is on same bus
     └─ Root Complex forwards back to Switch 0

4. PCIe Completion:
   TLP arrives at GPU 1 PCIe controller
     └─ Validates CRC
     └─ Checks destination address
     └─ Writes to local HBM

Key Differences from NVLink:
• Slower: 12 GB/s (PCIe Gen3) vs 20 GB/s (NVLink)
• Higher latency: ~1.5μs vs ~0.8μs
• Root Complex involvement in routing
• More protocol overhead (TLP headers)
• No hardware-accelerated atomic operations

ACS (Access Control Services) Issue:
────────────────────────────────────────

Problem: ACS can block P2P transfers
└─ ACS redirects all PCIe traffic through root complex
└─ Intended for security (prevents device-to-device DMA)
└─ Kills P2P performance (adds CPU involvement)

Solution: Disable ACS in BIOS
└─ BIOS → Advanced → PCIe Options → ACS = Disabled
└─ Allows direct device-to-device transfers
└─ Preserves P2P performance
```

---

## 4. MNNVL (Multi-Node NVLink) - P2P Across Nodes

```
When GPUs are in different nodes but connected via NVLink

Hardware Components:
┌──────────────────────────────────────────────────────────────┐
│                    Multi-Node Setup                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Node 0:                     NVLink Fabric                  │
│  ┌────────────────┐              ┌─────────────────┐         │
│  │ GPU 0 (H100)   │──NVLink 4──►│                 │         │
│  │ GPU 1 (H100)   │──┬────┬───►│   NVLink Switch │◄────────┤
│  └────────────────┘  │    │    │   (External)    │         │
│                      │    │    └────────┬────────┘         │
│  ┌────────────────┐  │    │             │                  │
│  │   CPU          │  │    │             └──────────────────┤
│  │   PCIe RC      │  │    │                                │
│  └────────────────┘  │    │                                │
│          │           │    │                                │
│          │ PCIe      │    │                                │
│          ▼           │    │    ┌─────────────────┐         │
│  ┌────────────────┐  │    └───►│                 │         │
│  │ NIC (Ethernet) │  │         │   NVLink Cable  │         │
│  └───┬────────────┘  │         │   (Optical)     │         │
│      │ IP Network    │         └────────┬────────┘         │
│      └───────────────┘                  │                  │
│                                         ▼                  │
│  Node 1:                  ┌────────────────────────────┐  │
│  ┌────────────────┐      │  Rack-Level NVLink Switch  │  │
│  │ GPU 2 (H100)   │◄─────┤  - 128 Ports               │  │
│  │ GPU 3 (H100)   │◄─────┤  - 57.6 TB/s bisection    │  │
│  └────────────────┘      └──────────────┬─────────────┘  │
│                                           │                │
│                                           └────────────────┤
└────────────────────────────────────────────────────────────┘

Data Transfer: GPU 0 (Node 0) → GPU 2 (Node 1)
────────────────────────────────────────

GPUs are in different nodes but appear as local via MNNVL!

1. GPU 0 Kernel thread writes to address 0x90000000:
   └─ GMMU translation: Virtual → Physical (Node 1 GPU 2)

2. Memory Hub detects remote address:
   └─ Routes to external NVLink switch

3. External NVLink Switch:
   └─ Receives NVLink packet from Node 0
   └─ Looks up destination (Node 1, GPU 2)
   └─ Routes through optical cables
   └─ Forwards to Node 1 NVLink switch

4. Node 1 NVLink Switch:
   └─ Receives packet
   └─ Routes to GPU 2
   └─ GPU 2 writes to local HBM

Performance:
└─ Latency: ~5-8μs (3x slower than local NVLink)
└─ Bandwidth: ~15-20 GB/s (80% of local NVLink)
└─ Still 5-10x faster than network!

Characteristics:
└─ Appears as single address space across nodes
└─ CUDA kernel doesn't know remote vs local
└─ Transparent to application
└─ cuMem API manages cross-node mappings
```

---

## 5. Page Table Management

### Virtual to Physical Translation

```
GPU Page Table Structure:
┌──────────────────────────────────────────────────────────────┐
│ GPU 0 Page Tables                                            │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│ Virtual Address: 0x7f3a00000000                              │
│                                                               │
│ [PDE] Page Directory Entry (L2)                              │
│ ┌─────────────────────────────────────────────┐              │
│ │ GPU: 0                                      │              │
│ │ Present: 1                                  │              │
│ │ Size: 2MB (huge page)                       │              │
│ │ PPN: 0x1a34000 (GPU 0 Physical)             │              │
│ └─────────────────────────────────────────────┘              │
│         │                                                    │
│         ▼                                                    │
│ [PTE] Page Table Entry (L1)                                  │
│ ┌─────────────────────────────────────────────┐              │
│ │ GPU: 1 (Peer!)                              │              │
│ │ Present: 1                                  │              │
│ │ Writable: 1                                 │              │
│ │ Cacheable: 1                                │              │
│ │ PPN: 0x7f2000 (GPU 1 Physical) ←────────────┼──────────┐   │
│ └─────────────────────────────────────────────┘          │   │
│         │                                                │   │
│         ▼                                                │   │
│ Translation:                                             │   │
│ Virtual 0x7f3a00000000 → GPU 1 Physical 0x7f2000        │   │
│                                                          │   │
└──────────────────────────────────────────────────────────┼───┘
                                                           │
                                   GPU 1 Page Tables      │
                                   don't need entry for  │
                                   GPU 1 local memory     │
                                                           │
                                   GPU 1 Physical Memory  │
                                   ┌────────────────────────▼──┐
                                   │ Address: 0x7f2000          │
                                   │ Data: [value to write]    │
                                   └────────────────────────────┘

Page Table Population:
────────────────────────

cudaDeviceEnablePeerAccess(peerGPU, 0);
  └─ CUDA driver adds entries to GPU 0's page tables
  └─ Maps peer GPU's physical address space
  └─ Typically uses 2MB huge pages for efficiency
  └─ Marks pages as "remote" for proper routing

cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream);
  └─ If dst is in peer GPU memory:
     └─ GMMU uses page tables to translate
     └─ Generates P2P write to peer
```

### Address Translation Cache (TLB)

```
GMMU Caching Hierarchy:

┌──────────────────────────────────────────────────────────────┐
│ L1 TLB (Per SM)                                             │
├──────────────────────────────────────────────────────────────┤
│ Size: 128-256 entries per SM                                │
│ Associativity: 4-way set associative                        │
│ Hit Latency: ~1 cycle                                       │
│ Miss Penalty: 10-20 cycles (L2 TLB lookup)                 │
│                                                               │
│ Entry Format:                                                 │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ VPN (Virtual Page Number)     │ 49 bits                 │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ PPN (Physical Page Number)    │ 28 bits                 │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ Flags (R/W/X/Cached/Remote)   │ 8 bits                  │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                               │
│ Behavior on P2P Access:                                       │
│ └─ First access: MISS → L2 lookup                           │
│ └─ L2 hit: ~15 cycles                                         │
│ └─ L2 miss: Page table walk (100+ cycles)                   │
│ └─ Subsequent accesses: L1/L2 hit, fast translation         │
└──────────────────────────────────────────────────────────────┘
         │
         │ On L1 Miss
         ▼
┌──────────────────────────────────────────────────────────────┐
│ L2 TLB (Shared across SMs)                                  │
├──────────────────────────────────────────────────────────────┤
│ Size: 2048-4096 entries (GPU-wide)                          │
│ Associativity: 8-way set associative                        │
│ Hit Latency: ~15 cycles                                     │
│ Miss Penalty: Page table walk (100-200 cycles)             │
│                                                               │
│ Services all SMs in the GPU                                 │
│ Includes remote (P2P) entries                               │
└──────────────────────────────────────────────────────────────┘
         │
         │ On L2 Miss
         ▼
┌──────────────────────────────────────────────────────────────┐
│ Page Table Walker (Hardware)                                │
├──────────────────────────────────────────────────────────────┤
│ Traverses GPU page tables in memory                           │
│                                                               │
│ Walk Process:                                                 │
│ └─ Load Page Directory Entry (PDE) from HBM                 │
│ └─ Check if entry present                                   │
│ └─ Load Page Table Entry (PTE) from HBM                     │
│ └─ Extract physical address                                  │
│ └─ Populate L2 TLB, then L1 TLB                             │
│                                                               │
│ Duration: 100-200 cycles for 2-level walk                   │
└──────────────────────────────────────────────────────────────┘

TLB Shootdown:
────────────────────────

Problem: What if memory mapping changes during access?

Process A: cudaFree(ptr)  while  Process B: accessing ptr

Solution:
1. cudaFree() invalidates page tables
2. Sends TLB shootdown interrupt to all SMs
3. All SMs flush L1 TLB entries
4. L2 TLB flushed atomically
5. If racing access occurs:
   └─ Page fault
   └─ Handler detects invalid mapping
   └─ Returns error (or segfault)

NCCL handles this by:
└─ Keeping buffers alive during communication
└─ Reference counting IPC mappings
└─ Coordinating free across ranks
```

---

## 6. Memory Controller and DRAM Details

```
HBM2/HBM2e Internal Organization:
┌──────────────────────────────────────────────────────────────┐
│ GPU Die                                                      │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                   GPU Core Logic                       │ │
│  │  (SMs, Cache, etc.)                                   │ │
│  └────────────────────────┬───────────────────────────────┘ │
│                           │                                  │
│  ┌────────────────────────▼──────────────────────────────┐   │
│  │          Memory Controller (8-12 controllers)        │   │
│  │                                                       │   │
│  │  ┌─────────┐  ┌─────────┐          ┌─────────┐     │   │
│  │  │  MC 0   │  │  MC 1   │   ...    │  MC 11  │     │   │
│  │  └───┬─────┘  └───┬─────┘          └───┬─────┘     │   │
│  │      │            │                  │             │   │
│  └──────┼────────────┼──────────────────┼─────────────┘   │
│         │            │                  │                 │
│ ┌───────▼────────────▼──────────────────▼────────────┐   │
│ │                                                      │   │
│ │          HBM2 Stack (On-package)                   │   │
│ │                                                      │   │
│ │  ┌────────┐  ┌────────┐         ┌────────┐        │   │
│ │  │ Layer 0│  │ Layer 1│  ...    │ Layer 7│        │   │
│ │  │ (DRAM) │  │ (DRAM) │         │ (DRAM) │        │   │
│ │  └────┬───┘  └───┬────┘         └───┬────┘        │   │
│ │       │          │                  │             │   │
│ │  ┌────▼──────────▼──────────────────▼──────────┐ │   │
│ │  │         Through-Silicon Via (TSV)          │ │   │
│ │  │   Vertical interconnect between layers    │ │   │
│ │  └───────────────────────────────────────────┘ │   │
│ │                                                │   │
│ └────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘

HBM2 Technical Specifications:
███████████████████████████████████████████████████████████████

Density:
└─ 8 Gb (1 GB) per die
└─ 4-high stack: 4 GB total
└─ 8-high stack: 8 GB total
└─ Modern: 16 GB, 32 GB, 64 GB stacks

Interface:
└─ 1024-bit wide bus per stack
└─ Up to 8 stacks per GPU
└─ 8192-bit total interface width

Bandwidth:
└─ HBM2: 256 GB/s per stack @ 2 Gbps
└─ HBM2e: 410 GB/s per stack @ 3.2 Gbps
└─ HBM3: 819 GB/s per stack @ 6.4 Gbps

Timing Parameters (HBM2):
└─ tCK: 0.5 ns (2 GHz)
└─ CL: 16 cycles (CAS latency)
└─ tRCD: 15 ns (RAS to CAS delay)
└─ tRP: 15 ns (RAS precharge)
└─ tRAS: 35 ns (RAS active time)

Read/Write Mechanism (HBM2):
┌────────────────────────────────────────────┐
│     Row Buffer (Sense Amplifiers)        │
│                                            │
│  ┌───────┐ ┌───────┐ ┌───────┐          │
│  │ Bank 0│ │ Bank 1│ │ Bank 2│   ...     │
│  │ Row X │ │ Row Y │ │ Row Z │          │
│  └───────┘ └───────┘ └───────┘          │
└────────────┬───────────────────────────────┘
             │
             │ Activate (Open Row)
             │
┌────────────▼──────────────────────────────┐
│  DRAM Array (Capacitors)                 │
│                                            │
│  Row X: [████████████████████]            │
│  Row Y: [████████████████████]            │
│  Row Z: [████████████████████]            │
│                                            │
│  Access transistor (per cell)             │
└───────────────────────────────────────────┘

Row Buffer Locality Optimization:
┌────────────────────────────────────────────┐
│     Scenario 1: Sequential Access        │
│                                            │
│  Addresses: 0x1000, 0x1004, 0x1008        │
│  └─ Same row (0x1000)                     │
│  └─ Row already open                       │
│  └─ Fast access: ~5 ns                    │
│                                            │
│     Scenario 2: Random Access            │
│                                            │
│  Addresses: 0x1000, 0x2000, 0x3000        │
│  └─ Different rows                         │
│  └─ Row activation needed each time        │
│  └─ Slower: ~30 ns (activate + access)    │
│                                            │
│ NCCL optimizes for:                        │
│ └─ Sequential writes (ring algorithm)     │
│ └─ High row buffer locality               │
└────────────────────────────────────────────┘
```

---

## 7. Complete Data Flow Summary

### PCIe P2P Transfer (Detailed Timeline)

```
Transfer: 4KB from GPU 0 to GPU 1 (32 threads × 128 bytes)

Time (ns) │ Event
──────────┼────────────────────────────────────────────────────────────────────
0         │ CPU launches kernel on GPU 0
5         │ Command processor fetches command
10        │ Kernel dispatches to SMs
15        │ Warp 0 Thread 0 generates write to GPU 1
18        │ GMMU L1 TLB miss
25        │ GMMU L2 TLB miss
50        │ Page table walk (100 cycles @ 2 GHz = 50 ns)
100       │ Translation cached in L2 TLB
105       │ Translation cached in L1 TLB
108       │ GPU 0 memory hub receives write request
112       │ PCIe controller formats TLP (header)
118       │ TLP enters PCIe switch
125       │ TLP forwarded to root complex (if same root)
130       │ TLP forwarded back through switch
138       │ TLP arrives at GPU 1 PCIe PHY
142       │ TLP decoded in GPU 1 controller
146       │ Request sent to GPU 1 memory hub
152       │ GMMU translation (likely hit in L2)
162       │ Write sent to HBM2 channel 3
172       │ DRAM controllers receive write
182       │ Row activation (tRCD = 15 ns = 30 cycles)
212       │ Data written to sense amplifiers
222       │ Row precharge begins (tRP = 15 ns)
252       │ Write complete in GPU 1 HBM

Total: ~250 ns per 4KB write
Effective bandwidth: 4KB / 250ns = 16 GB/s (includes overhead)

Multiple warps execute in parallel:
└─ GPU has ~100-200 SMs
└─ Each SM can run 16-32 warps
└─ Hundreds of concurrent writes
└─ Full PCIe bandwidth: 12-24 GB/s
```

### NVLink P2P Transfer (Detailed Timeline)

```
Transfer: 4KB from GPU 0 to GPU 1 (direct NVLink)

Time (ns) │ Event
──────────┼────────────────────────────────────────────────────────────────────
0         │ CPU launches kernel on GPU 0
5         │ Command processor fetches command
10        │ Kernel dispatches to SMs
15        │ Warp 0 Thread 0 generates write to GPU 1
18        │ GMMU L1 TLB miss
25        │ GMMU L2 TLB miss
50        │ Page table walk (50 ns)
100       │ Translation cached in TLBs
105       │ GPU 0 memory hub receives write request
108       │ Packet formatted for NVLink (header + data)
112       │ NVLink PHY begins transmission (8b/10b encode)
118       │ Data on NVLink wire (3 ns per hop)
121       │ Arrives at GPU 1 NVLink PHY
124       │ NVLink PHY decodes packet
128       │ Request sent to GPU 1 memory hub
134       │ GMMU translation (hits in L2)
144       │ Write sent to HBM2 channel
154       │ DRAM controllers receive write
164       │ Row activation (tRCD = 15 ns)
194       │ Data written to sense amplifiers
204       │ Row precharge (tRP = 15 ns)
234       │ Write complete in GPU 1 HBM

Total: ~234 ns per 4KB write

Key Differences from PCIe:
└─ Faster encode/decode (no PCIe protocol overhead)
└─ Direct GPU-to-GPU (no root complex)
└─ Lower latency per hop (3ns vs 10-20ns)
└─ Full-duplex simultaneous transfers
```

### Bandwidth Factors

```
Achievable P2P Bandwidth (PCIe):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Maximum Theoretical:
┌────────────────────────────────────────────┐
│ PCIe Gen3 x16: 15.75 GB/s                 │
│ PCIe Gen4 x16: 31.5 GB/s                  │
│ PCIe Gen5 x16: 63.0 GB/s                  │
└────────────────────────────────────────────┘

Real-World Achievable:
┌────────────────────────────────────────────┐
│ Protocol overhead: ~5-10%                  │
│ └─ TLP headers, link layer overhead       │
│                                            │
│ Flow control: ~2-3%                        │
│ └─ DLLPs (Data Link Layer Packets)        │
│                                            │
│ Transaction ordering: ~1-2%                │
│ └─ Posted vs non-posted transactions      │
│                                            │
│ Efficiency: ~85-90%                        │
│                                            │
│ PCIe Gen3 x16: ~12 GB/s (real)            │
│ PCIe Gen4 x16: ~24 GB/s (real)            │
│ PCIe Gen5 x16: ~48 GB/s (real)            │
└────────────────────────────────────────────┘

Achievable P2P Bandwidth (NVLink):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NVLink 3.0 (A100):
┌────────────────────────────────────────────┐
│ Theoretical: 50 GB/s per link (bidir)     │
│ Bidirectional: 25 GB/s each direction     │
│ 12 Links: 300 GB/s total                  │
│                                            │
│ Real-world single link: ~45 GB/s          │
│ └─ Encoding overhead (8b/10b)             │
│ └─ Protocol overhead                      │
│                                            │
│ Effective per GPU: 20 GB/s (saturating)   │
└────────────────────────────────────────────┘

NVLink 4.0 (H100):
┌────────────────────────────────────────────┐
│ Theoretical: 90 GB/s per link (bidir)     │
│ Bidirectional: 45 GB/s each direction     │
│ 18 Links: 810 GB/s total                  │
│                                            │
│ Real-world: ~40 GB/s per GPU (sustained)  │
└────────────────────────────────────────────┘
```

---

## 8. Hardware-Software Interaction

### NCCL to Hardware Mapping

```
NCCL Concept → Hardware Implementation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NCCL Channel                    → Set of GPU P2P connections
                                  └─ Pre-established memory mappings

NCCL Buffer (sendbuff)          → GPU 0 HBM physical pages
                                  └─ Exported via IPC handles

NCCL Connector                  → P2P connection state
                                  └─ Remote memory addresses
                                  └─ Flags (read/write/etc)

NCCL Kernel (device code)       → CUDA kernel on GPU SMs
                                  └─ Warps execute primitives
                                  └─ Generate P2P memory ops

Primitives (directSend)         → GPU core instructions
                                  └─ LDG (load global)
                                  └─ STG (store global)
                                  └─ LDGDEPBAR (dependency)

nvmlDeviceGetNvLinkState()      → HW query NVLink PHY
                                  └─ Reads NVLink registers
                                  └─ Returns link status

cudaDeviceEnablePeerAccess()    → GPU MMIO write
                                  └─ Updates GMMU page tables
                                  └─ Enables hardware routing
```

### BAR (Base Address Register) Access

```
How GPUs are mapped into CPU address space:

lspci -vvv -s 01:00.0  # Shows GPU 0 PCI config

┌────────────────────────────────────────────────────────┐
│ 01:00.0 VGA compatible controller: NVIDIA Corporation  │
│ ...                                                    │
│ Memory at 7f3800000000 (64-bit, prefetchable) [size=16G]
│ Memory at 7f3c00000000 (64-bit, prefetchable) [size=32M]
│ Memory at 7f3c02000000 (64-bit, prefetchable) [size=32M]
└────────────────────────────────────────────────────────┘

These BARs are mapped by BIOS/UEFI and kernel:
• BAR0: GPU configuration registers (MMIO)
• BAR1: GPU memory (VRAM) aperture
• BAR2: GPU IO space

When CPU accesses 0x7f3800000000:
  └─ PCIe controller converts to GPU config cycle
  └─ GPU responds with register contents

When GPU accesses peer GPU memory:
  └─ Uses physical address (not BAR)
  └─ No CPU involvement
  └─ Direct GPU-to-GPU
```

---

## Summary

**Hardware Components Involved in P2P:**
1. GPU Cores (SMs) - Execute kernel, generate memory requests
2. GMMU (GPU MMU) - Virtual to physical translation
3. TLB Caches (L1/L2) - Accelerate translations
4. GPU Memory Hub - Routes requests locally or remotely
5. NVLink PHY - Physical transceiver (if NVLink)
6. NVLink Switch - Crossbar for multi-GPU systems
7. PCIe PHY - Physical transceiver (if PCIe)
8. PCIe Root Complex/Switch - Routing (PCIe only)
9. HBM2/HBM2e DRAM - Stores data in GPU memory
10. Memory Controller - Manages DRAM access

**Data Flow Stages:**
1. Kernel generates virtual address
2. GMMU translates to physical address
3. Memory Hub routes based on address
4. Protocol encoding (NVLink or PCIe)
5. Physical transmission over interconnect
6. Receiver decodes packet
7. Local GMMU translation at destination
8. DRAM controller writes to HBM

**Latency Breakdown (NVLink):**
- Address translation: 50-100 ns
- Header generation: 10 ns
- Transmission: 3-10 ns
- Reception/decode: 20 ns
- DRAM write: 150-200 ns
- Total: ~250 ns minimum

**Bandwidth Limits:**
- PCIe Gen3: ~12 GB/s
- PCIe Gen4: ~24 GB/s
- PCIe Gen5: ~48 GB/s
- NVLink 3.0: ~20 GB/s (per direction)
- NVLink 4.0: ~40 GB/s (per direction)

The magic of P2P is that **all of this happens in hardware** - the CPU is not involved except for initial setup!
