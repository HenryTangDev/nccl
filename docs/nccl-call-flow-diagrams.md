# NCCL Call Flow UML Sequence Diagrams

This document provides UML sequence diagrams for NCCL collective communication operations, illustrating the complete call flow from user API to kernel execution and network operations.

## Overview

NCCL collective operations follow a consistent pattern:
1. **API Layer**: User calls public NCCL API (e.g., `ncclAllReduce`)
2. **Enqueue Layer**: Validation and task preparation (`ncclEnqueueCheck`)
3. **Planning Layer**: Algorithm selection and resource allocation
4. **Launch Layer**: Kernel launch preparation and execution
5. **Device Layer**: GPU kernel execution with primitives
6. **Proxy Layer**: Network operations via proxy threads

---

## AllReduce Call Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant API as ncclAllReduce<br/>(collectives.cc:93)
    participant Enqueue as ncclEnqueueCheck<br/>(enqueue.cc)
    participant Planner as ncclTasksRegAndEnqueue<br/>(enqueue.cc:264)
    participant Launch as ncclLaunchKernel<br/>(enqueue.cc)
    participant Device as GPU Kernel<br/>(device/all_reduce.h)
    participant Proxy as Proxy Thread<br/>(proxy.cc)
    participant Network as Transport Layer<br/>(transport/)

    User->>API: ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)
    
    Note over API: Create ncclInfo struct with<br/>ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS
    API->>API: struct ncclInfo info = {<br/>  ncclFuncAllReduce, "AllReduce",<br/>  sendbuff, recvbuff, count, datatype, op, 0,<br/>  comm, stream, ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS<br/>}
    
    API->>Enqueue: ncclEnqueueCheck(&info)
    
    Note over Enqueue: Argument validation and<br/>task enqueueing
    Enqueue->>Enqueue: Validate arguments
    Enqueue->>Enqueue: Create ncclTaskColl
    Enqueue->>Enqueue: Add to comm->planner.collTaskQueue
    
    Note over Enqueue: Return to user - operation is async
    Enqueue-->>API: ncclSuccess
    API-->>User: ncclSuccess
    
    Note over User, Network: === Asynchronous Execution Phase ===
    
    Note over Planner: During stream synchronization or<br/>next collective operation
    Planner->>Planner: ncclPrepareTasks(comm, ...)
    
    Note over Planner: Algorithm Selection Phase
    Planner->>Planner: getAlgoInfo()<br/>- Choose RING vs TREE<br/>- Select protocol (LL/LL128/Simple)<br/>- Determine nChannels
    
    Note over Planner: Resource Registration
    Planner->>Planner: ncclRegisterCollBuffers()
    Planner->>Planner: Create ncclDevWorkColl struct
    
    Note over Launch: Kernel Launch Phase
    Launch->>Launch: ncclLaunchKernelBefore_NoUncapturedCuda()
    Launch->>Launch: Set up kernel arguments<br/>channelMask, workStorageType
    Launch->>Device: cudaLaunchKernel<br/>(nChannels, 1, 1) grid<br/>(nThreads, 1, 1) block
    
    Note over Device: GPU Kernel Execution
    Device->>Device: Load comm metadata (warp 0)<br/>Load channel data (warp 1)
    
    alt Ring Algorithm
        Device->>Device: runRing<T, RedOp, Proto>()
        Note over Device: REDUCE-SCATTER PHASE<br/>(k-1 steps)
        Device->>Device: Step 0: directSend()
        loop k-2 iterations
            Device->>Device: directRecvReduceDirectSend()
        end
        Device->>Device: Step k-1: directRecvReduceCopyDirectSend()
        
        Note over Device: ALL-GATHER PHASE<br/>(k-1 steps)  
        loop k-2 iterations
            Device->>Device: directRecvCopyDirectSend()
        end
        Device->>Device: Final: directRecv()
        
    else Tree Algorithm
        Device->>Device: runTreeUpDown<T, RedOp, Proto>()
        Note over Device: REDUCE PHASE (fan-in)
        alt Root Node
            Device->>Device: directRecvReduceCopy()
        else Leaf Node  
            Device->>Device: directSend()
        else Middle Node
            Device->>Device: directRecvReduceDirectSend()
        end
        
        Note over Device: BROADCAST PHASE (fan-out)
        alt Root Node
            Device->>Device: directSendFromOutput()
        else Leaf Node
            Device->>Device: directRecv()
        else Middle Node
            Device->>Device: directRecvCopyDirectSend()
        end
    end
    
    Note over Device, Network: Inter-GPU Communication
    par P2P Communication
        Device->>Network: P2P transfers<br/>(via P2P_DIRECT or FIFO)
        Network->>Network: NVLink/PCIe data movement
    and Network Communication  
        Device->>Proxy: Notify proxy thread<br/>(via ncclConnFifo)
        Proxy->>Network: InfiniBand/TCP send/recv
        Network->>Proxy: Complete network operation
        Proxy->>Device: Signal completion
    end
    
    Note over Launch: Post-kernel operations
    Launch->>Launch: ncclLaunchKernelAfter_NoCuda()
    Launch->>Launch: Update operation counters
```

---

## AllGather Call Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant API as ncclAllGather<br/>(collectives.cc:79)
    participant Enqueue as ncclEnqueueCheck
    participant Device as GPU Kernel<br/>(device/all_gather.h)
    participant Network as Transport Layer

    User->>API: ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream)
    
    API->>API: struct ncclInfo info = {<br/>  ncclFuncAllGather, "AllGather",<br/>  sendbuff, recvbuff, sendcount, datatype, ncclSum, 0,<br/>  comm, stream, ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS<br/>}
    
    API->>Enqueue: ncclEnqueueCheck(&info)
    Enqueue-->>API: ncclSuccess (async)
    API-->>User: ncclSuccess
    
    Note over Device: Ring AllGather Algorithm<br/>(k-1 steps total)
    
    alt Step 0 (Initial)
        alt In-place operation
            Device->>Device: send (local data already in position)
        else Out-of-place
            Device->>Device: copySend (copy local data to output buffer)
        end
    end
    
    loop Steps 1 to k-2
        Device->>Device: recvCopySend<br/>(receive block, copy to output buffer, forward)
        Device->>Network: Forward data to next GPU in ring
    end
    
    Note over Device: Final step (k-1)
    Device->>Network: recv (receive final missing block)
    
    Note over Device: Result: All GPUs have complete data from all ranks
```

---

## Broadcast Call Flow Diagram  

```mermaid
sequenceDiagram
    participant User
    participant API as ncclBroadcast<br/>(collectives.cc:106)
    participant Enqueue as ncclEnqueueCheck
    participant Device as GPU Kernel<br/>(device/broadcast.h)
    participant Network as Transport Layer

    User->>API: ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream)
    
    API->>API: struct ncclInfo info = {<br/>  ncclFuncBroadcast, "Broadcast",<br/>  sendbuff, recvbuff, count, datatype, ncclSum, root,<br/>  comm, stream, BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS<br/>}
    
    API->>Enqueue: ncclEnqueueCheck(&info)
    Enqueue-->>API: ncclSuccess (async)
    API-->>User: ncclSuccess
    
    Note over Device: Ring Broadcast Algorithm<br/>(Chain pattern from root)
    
    alt Root GPU
        alt In-place operation
            Device->>Network: send (data to next GPU in chain)
        else Out-of-place  
            Device->>Device: copySend (copy to recvbuff, send to next)
        end
    else Middle GPUs
        Device->>Network: recv (from previous GPU)
        Device->>Device: copy to local recvbuff
        Device->>Network: send (forward to next GPU)
    else Last GPU
        Device->>Network: recv (from previous GPU)
        Device->>Device: copy to local recvbuff
    end
    
    Note over Device: Result: All GPUs have root's data
```

---

## Reduce Call Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant API as ncclReduce<br/>(collectives.cc:126)
    participant Enqueue as ncclEnqueueCheck
    participant Device as GPU Kernel<br/>(device/reduce.h)
    participant Network as Transport Layer

    User->>API: ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream)
    
    API->>API: struct ncclInfo info = {<br/>  ncclFuncReduce, "Reduce",<br/>  sendbuff, recvbuff, count, datatype, op, root,<br/>  comm, stream, REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS<br/>}
    
    API->>Enqueue: ncclEnqueueCheck(&info)
    Enqueue-->>API: ncclSuccess (async)
    API-->>User: ncclSuccess
    
    Note over Device: Ring Reduce Algorithm<br/>(Chain pattern toward root)
    
    alt First GPU (initiator)
        Device->>Network: send (local data to next GPU)
    else Middle GPUs
        Device->>Network: recv (partially reduced data)
        Device->>Device: reduce with local data
        Device->>Network: send (updated result to next GPU)
    else Root GPU
        Device->>Network: recv (final partial result)
        Device->>Device: recvReduceCopy<br/>(final reduction + copy to recvbuff)
    end
    
    Note over Device: Result: Root GPU has reduced result
```

---

## ReduceScatter Call Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant API as ncclReduceScatter<br/>(collectives.cc:139)
    participant Enqueue as ncclEnqueueCheck
    participant Device as GPU Kernel<br/>(device/reduce_scatter.h)
    participant Network as Transport Layer

    User->>API: ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream)
    
    API->>API: struct ncclInfo info = {<br/>  ncclFuncReduceScatter, "ReduceScatter",<br/>  sendbuff, recvbuff, recvcount, datatype, op, 0,<br/>  comm, stream, REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS<br/>}
    
    API->>Enqueue: ncclEnqueueCheck(&info)
    Enqueue-->>API: ncclSuccess (async)
    API-->>User: ncclSuccess
    
    Note over Device: Ring ReduceScatter Algorithm<br/>(k-1 steps, like AllReduce reduce-scatter phase)
    
    Note over Device: Step 0: Initial send
    Device->>Network: send (one data block to next GPU)
    
    loop Steps 1 to k-2
        Device->>Network: recv (partially reduced block from previous GPU)
        Device->>Device: reduce with corresponding local block
        Device->>Network: send (reduced result to next GPU)  
    end
    
    Note over Device: Final step (k-1)
    Device->>Network: recv (final partial block)
    Device->>Device: recvReduceCopy<br/>(final reduction + copy to recvbuff)
    
    Note over Device: Result: Each GPU has reduced result for its assigned segment
```

---

## Point-to-Point Communication Diagram

```mermaid
sequenceDiagram
    participant Sender as GPU A<br/>(ncclSend)
    participant RecvAPI as GPU B<br/>(ncclRecv)  
    participant SendEnqueue as Send Enqueue
    participant RecvEnqueue as Recv Enqueue
    participant Device as GPU Kernels
    participant Network as P2P/Network Transport

    Note over Sender, RecvAPI: Symmetric Send/Recv setup
    
    Sender->>SendEnqueue: ncclSend(sendbuff, count, datatype, peer, comm, stream)
    RecvAPI->>RecvEnqueue: ncclRecv(recvbuff, count, datatype, peer, comm, stream)
    
    SendEnqueue->>SendEnqueue: struct ncclInfo info = { ncclFuncSend, "Send",<br/>  NULL, sendbuff, count, datatype, ncclSum, peer,<br/>  comm, stream, 1, 1 }
    
    RecvEnqueue->>RecvEnqueue: struct ncclInfo info = { ncclFuncRecv, "Recv",<br/>  NULL, recvbuff, count, datatype, ncclSum, peer,<br/>  comm, stream, 1, 1 }
    
    SendEnqueue-->>Sender: ncclSuccess (async)
    RecvEnqueue-->>RecvAPI: ncclSuccess (async)
    
    Note over Device: P2P Kernel Execution (coordinated)
    
    par Send Side
        Device->>Network: directSend() or send via transport
    and Recv Side  
        Device->>Network: directRecv() or recv via transport
    end
    
    alt Intra-node (P2P_DIRECT)
        Network->>Network: Direct GPU memory access<br/>(NVLink/PCIe)
    else Inter-node
        Network->>Network: InfiniBand/TCP network transfer
    end
    
    Note over Device: Transfer completion
```

---

## Group Operations Call Flow

```mermaid
sequenceDiagram
    participant User
    participant GroupAPI as Group API<br/>(group.cc)
    participant Multiple as Multiple Ops
    participant Launch as Kernel Launch<br/>(enqueue.cc)
    participant Device as GPU Kernels

    User->>GroupAPI: ncclGroupStart()
    Note over GroupAPI: Set comm->groupJob = ncclGroupJobRunning

    User->>Multiple: ncclAllReduce(...)
    User->>Multiple: ncclBroadcast(...)  
    User->>Multiple: ncclSend(...) / ncclRecv(...)
    
    Note over Multiple: Operations enqueued but not launched
    Multiple->>Multiple: Add to planner queues
    
    User->>GroupAPI: ncclGroupEnd()
    
    Note over GroupAPI: Launch all grouped operations
    GroupAPI->>GroupAPI: ncclGroupJobMain()
    
    loop For each communicator with pending work
        GroupAPI->>Launch: ncclLaunchPrepare(comm)
        GroupAPI->>Launch: ncclLaunchKernelBefore_NoUncapturedCuda(comm, plan)
        GroupAPI->>Launch: ncclLaunchKernel(comm, plan)
        GroupAPI->>Launch: ncclLaunchKernelAfter_NoCuda(comm, plan)
        GroupAPI->>Launch: ncclLaunchFinish(comm)
    end
    
    Note over Device: All operations execute in parallel/overlapped
    Device->>Device: Multiple collective kernels
    Device->>Device: P2P operations
    
    GroupAPI-->>User: ncclSuccess
```

---

## Key Implementation Notes

### Common Flow Pattern
All collective operations follow the same basic pattern:
1. **API validation** and `ncclInfo` struct creation
2. **Enqueue** via `ncclEnqueueCheck` (returns immediately)
3. **Asynchronous execution** triggered by stream sync or next operation
4. **Algorithm selection** and resource planning
5. **Kernel launch** with specialized device functions
6. **Transport-specific** communication (P2P, network, shared memory)

### Critical Constants from Source
- `ALLREDUCE_CHUNKSTEPS = NCCL_STEPS/2 = 4`
- `ALLREDUCE_SLICESTEPS = NCCL_STEPS/4 = 2`  
- `NCCL_STEPS = 8` (pipeline depth)
- Grid dimension: `(nChannels, 1, 1)`
- Block dimension: `(threadPerBlock, 1, 1)`

### Transport Integration
- **P2P_DIRECT**: Direct GPU-to-GPU within same process
- **P2P via FIFO**: GPU→FIFO→GPU with intermediate buffering  
- **Network**: GPU→Proxy Thread→Network→Remote Proxy→Remote GPU
- **Shared Memory**: GPU→CPU→Shared Memory→Remote CPU→Remote GPU

### Algorithm Selection Logic
The diagrams show simplified algorithm selection. Actual selection considers:
- Message size thresholds
- Available hardware (NVLink, InfiniBand)  
- Topology (intra-node vs inter-node)
- Protocol capabilities (LL128 requirements)
- Performance tuning parameters

These diagrams provide the complete call flow understanding needed for debugging, optimization, and extending NCCL collective operations.