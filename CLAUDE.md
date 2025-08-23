# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NCCL (pronounced "Nickel") is NVIDIA's library of optimized primitives for inter-GPU communication. It implements collective operations like all-reduce, all-gather, reduce, broadcast, and reduce-scatter, optimized for multi-GPU systems using PCIe, NVLink, NVswitch, and various network interconnects.

## Build System

### Primary Build Commands
```bash
# Build the library (most common)
make -j src.build

# Build with custom CUDA path
make src.build CUDA_HOME=/path/to/cuda

# Build for specific GPU architecture (faster compilation)
make -j src.build NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

# Install system packages
make pkg.debian.build  # For Debian/Ubuntu
make pkg.redhat.build  # For RedHat/CentOS
make pkg.txz.build     # OS-agnostic tarball
```

### Build Configuration
- **BUILDDIR**: Build output directory (default: `./build/`)
- **DEBUG**: Set to 1 for debug builds with `-O0 -G -g`
- **VERBOSE**: Set to 1 for verbose compilation output
- **CUDA_HOME**: CUDA installation path (default: `/usr/local/cuda`)
- **NVCC_GENCODE**: GPU architectures to compile for (auto-detected from CUDA version)

### Key Build Artifacts
- `build/lib/libnccl.so.X.Y.Z` - Main shared library
- `build/lib/libnccl_static.a` - Static library
- `build/include/nccl.h` - Public API header
- `build/bin/ncclras` - RAS (Reliability, Availability, Serviceability) client

## Architecture Overview

### Core Components
- **src/collectives.cc** - Main collective operation implementations
- **src/transport/** - Communication transport layer (IB, TCP, shared memory, P2P)
- **src/device/** - GPU kernel implementations for collective operations
- **src/graph/** - Topology detection and optimization (rings, trees, tuning)
- **src/plugin/** - Plugin system for network, profiler, and tuner extensions

### Communication Protocols

NCCL employs three protocols optimized for different bandwidth/latency trade-offs:

1. **Simple Protocol**:
   - **Goal**: High bandwidth (near peak)
   - **Latency**: ~6μs per hop
   - **Synchronization**: Memory fences (high overhead)
   - **Use case**: Large message transfers

2. **LL (Low Latency) Protocol**:
   - **Goal**: Low latency (~1μs per hop)
   - **Bandwidth**: 25-50% of peak
   - **Format**: 4B data + 4B flag using 8-byte atomic operations
   - **Limitation**: Forces intermediate buffer in host memory, prevents GPU Direct RDMA

3. **LL128 Protocol**:
   - **Goal**: Low latency (~2μs per hop) + high bandwidth (~95% of peak)
   - **Format**: 120B data + 8B flag
   - **Requirements**: Atomic 128-byte writes (not supported on all systems)
   - **Advantage**: Combines benefits of Simple and LL protocols

### Transport Layer Architecture

**Transport Selection Logic** (`src/transport.cc`):
```
Transport Priority: P2P > SHM > NET > COLLNET > PROFILER
```

**Transport Types:**
1. **P2P Transport** (`transport/p2p.cc`):
   - **P2P_DIRECT**: Direct GPU-to-GPU transfers without intermediate FIFO buffer
   - **GPUDirect P2P**: NVLink or PCIe-based GPU memory access
   - **IPC/CUMEM modes**: Inter-process GPU memory sharing

2. **Network Transport** (`transport/net_ib.cc`, `transport/net_socket.cc`):
   - **InfiniBand**: RDMA-based with GPUDirect RDMA optimization
   - **TCP Sockets**: Fallback network transport
   - **Multi-channel**: 2 logical channels per remote GPU for bandwidth optimization

3. **Shared Memory** (`transport/shm.cc`):
   - Intra-node communication via shared memory segments
   - Used when P2P is suboptimal (e.g., inter-socket PCIe traffic)
   - Lock-free ring buffers for producer-consumer patterns

### Communication Channel Management

**Channel Organization:**
- NCCL subdivides collectives into multiple **communication channels** (CUDA blocks)
- Each channel runs on separate SM to avoid bottlenecks and increase parallelism
- Grid dimension: `(nChannels, 1, 1)` with one-to-one channel-to-block mapping
- Data partitioning: Total data split across channels, then into loop iterations, then chunks

**Pipeline Execution:**
- Each channel buffer divided into **8 slots** (NCCL_STEPS parameter)
- Pipeline stages: data transfer, computation, and network operations overlap
- **chunkCount** elements processed per pipeline step
- **loopCount** iterations when data exceeds buffer capacity

**CUDA Hierarchy Mapping:**
- **Grid**: One block per communication channel
- **Block**: Variable threads (NCCL_MIN_NTHREADS to NCCL_MAX_NTHREADS)
- **Warps**: Specialized roles - warp 0 (metadata), warp 1 (channel data), others (work)
- **Threads**: Process multiple elements with vectorized operations

### Collective Algorithm Patterns

**Non-pipelined Algorithms** (must complete iteration before starting next):
- **Ring AllReduce**: 2k-1 steps (reduce-scatter + all-gather phases)
- **Ring AllGather**: k-1 steps (copy and forward data blocks)
- **Ring ReduceScatter**: k-1 steps (reduce and scatter unique segments)

**Pipelined Algorithms** (can overlap consecutive iterations):
- **Tree AllReduce**: Reduce-up then broadcast-down phases
- **Ring Broadcast**: Chain-based data dissemination from root
- **Ring Reduce**: Chain-based reduction toward root

### Device Code Generation System

**Code Generation Pipeline** (`src/device/generate.py`):
1. **Template Instantiation**: Creates specialized kernels for each (collective, reduction_op, datatype, algorithm, protocol) tuple
2. **Kernel Specialization**: Generates optimized kernels using template-based approach
3. **CUDA Architecture Filtering**: Only generates kernels supported by target GPU architectures

**Communication Primitives:**
- **send/recv**: Basic data transfer operations
- **recvReduceSend**: Receive, reduce with local data, forward result
- **recvCopySend**: Receive, copy to output buffer, forward unchanged
- **directSend/directRecv**: P2P_DIRECT mode without intermediate buffers

## Testing

Tests are maintained in a separate repository:
```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make
./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
```

### Plugin Testing
For tuner plugin tests:
```bash
cd ext-tuner/example/test
make test
```

## Development Workflow

### Advanced Build Options
```bash
# Debug build with tracing
make DEBUG=1 VERBOSE=1 TRACE=1

# Disable NVTX profiling markers
make NVTX=0

# Address sanitizer build
make ASAN=1

# Code coverage build
make GCOV=1 DEBUG=1
```

### Debugging and Profiling
**Logging Configuration:**
- `NCCL_DEBUG=INFO` - Detailed operation logging
- `NCCL_DEBUG_SUBSYS=COLL,P2P,NET` - Subsystem-specific debug output
- `NCCL_DEBUG_FILE=debug.log` - Log to file instead of stderr

**Performance Analysis:**
- `NCCL_ALGO` - Force algorithm: `TREE`, `RING`, `COLLNET_DIRECT`, `NVLS`
- `NCCL_PROTO` - Force protocol: `LL`, `LL128`, `SIMPLE`  
- `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` - Control channel parallelism

**Algorithm Selection Guidelines:**
- **Ring algorithms**: Excel for large messages (bandwidth-optimal)
- **Tree algorithms**: Best for small messages (latency-optimal)  
- **Protocol selection**: LL/LL128 for small messages, Simple for large transfers
- **Inter-node vs Intra-node**: Different protocols perform better based on transport type

**Channel Buffer Sizes (default configuration):**
- Simple: 4 MiB total, 512 KiB per slot
- LL: 256 KiB total, 32 KiB per slot (16 KiB effective data)
- LL128: ~4800 KiB total, 600 KiB per slot (562.5 KiB effective data)

## Plugin Architecture

NCCL supports three types of plugins:
- **Network plugins** (`ext-net/`) - Custom network transport implementations
- **Profiler plugins** (`ext-profiler/`) - Performance profiling and monitoring hooks
- **Tuner plugins** (`ext-tuner/`) - Algorithm and parameter tuning based on measured performance

## Version Management

Current version is defined in `makefiles/version.mk`:
- NCCL_MAJOR: 2
- NCCL_MINOR: 27  
- NCCL_PATCH: 7

Version information is embedded in build artifacts through template processing of `src/nccl.h.in`.

## Code Organization

### Header Files
- **src/nccl.h.in** - Template for public API header (processed during build)
- **src/include/** - Internal headers and device-side API definitions

### Transport Layer
- **net_ib.cc** - InfiniBand RDMA transport
- **net_socket.cc** - TCP/IP socket transport  
- **shm.cc** - Shared memory transport
- **p2p.cc** - GPU peer-to-peer transport

### Memory Management
- **src/allocator.cc** - Custom memory allocator
- **src/register/** - Memory registration for RDMA operations

## Common File Patterns

- `.cc` files - C++ source code (host-side implementation)
- `.cu` files - CUDA source code (device-side kernels)  
- `.h` files in `src/include/` - Internal headers
- `Makefile` hierarchy - Recursive build system with common configuration in `makefiles/`
- `*.in` files - Templates processed during build (version substitution)
- `generate.py` - Python scripts for automated code generation
- `*wrap.h` - Dynamic library loading wrappers (e.g., `ibvwrap.h`, `cudawrap.h`)

## Performance Characteristics

**Protocol Performance Trade-offs:**
- **Simple**: Near-peak bandwidth, ~6μs latency, best for large messages
- **LL**: ~1μs latency, 25-50% bandwidth, best for small inter-node transfers  
- **LL128**: ~2μs latency, ~95% bandwidth, excellent over NVLink for all sizes

**Collective Algorithm Optimization:**
- **Element Definition**: Bytes for AllGather/Broadcast, data types for reduction ops
- **Ring AllReduce**: Bandwidth-optimal with 2*(N-1) communication steps
- **Tree AllReduce**: Latency-optimal with 2*log(N) steps, asymmetric warp allocation
- **Pipeline Overlap**: Non-pipelined algorithms complete iterations sequentially
- **Multi-level Parallelism**: Channel, slot, warp, and thread-level concurrent execution

**Transport Optimization Features:**
- **P2P_DIRECT**: Eliminates intermediate FIFO buffer copies within same process
- **GPUDirect RDMA**: Direct NIC-to-GPU memory access when on same PCIe switch
- **Multi-channel QPs**: 2 logical channels per peer with round-robin traffic splitting  
- **Local Flush**: Loop-back RDMA_READ ensures PCIe write completion for data consistency