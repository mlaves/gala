# Educational ML Inference Library — Architecture Design Document

## Overview

This document describes the architecture of an educational, multi-backend ML inference library written in Swift. The primary development target is Apple Silicon, but the design is explicitly portable to Linux (x86) and future CUDA or ANE targets. The library is designed with autograd as a first-class architectural concern, even though the autograd engine is not implemented in the initial version.

---

## Goals

- Educational clarity: every layer is independently understandable and testable
- Multi-backend dispatch with clean abstraction boundaries
- Lazy evaluation with an eager-feeling public API
- Autograd-ready graph representation from day one
- Portable to Linux without conditional compilation leaking across module boundaries
- Raw Metal compute shaders (no MPSGraph) for full visibility into GPU execution

---

## Non-Goals (Initial Version)

- Autograd implementation (architecture must support it, not implement it)
- Training
- CUDA backend (stub only)
- ANE backend (stub only)
- Operator fusion passes (structure is present, passes are not)

---

## Evaluation Strategy: Lazy with Eager Feel

The library uses **lazy (deferred) evaluation**. Calling an operator like `a + b` does not execute anything — it appends a node to the computation graph. Execution is deferred until the value is explicitly needed (inspected, printed, passed to a loss, etc.), at which point the graph is realized automatically.

This approach is chosen over eager evaluation for the following reasons:

- The lazy graph is also the autograd tape — no second data structure is needed later
- Operator fusion is possible because the scheduler sees the full subgraph before committing to execution
- MPSGraph, Metal, and future backends are all graph-oriented natively
- Eager mode in PyTorch required a full re-architecture (`torch.compile`) to recover these properties — this library avoids that regret

The API feels eager because materialization is automatic and transparent to the user. This is the same model used by MLX and JAX.

---

## Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Public Swift API                    │
│         Tensor, operator overloads, device literals  │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│              Computation Graph (IR)                  │
│     GraphNode { op, inputs, shape, dtype, device }   │
│         Lazy — nothing executes until realized       │
└──────┬──────────────┬──────────────┬────────────────┘
       │              │              │
┌──────▼──────┐ ┌─────▼──────┐ ┌────▼────────────────┐
│  Scheduler  │ │  Optimizer  │ │  Device Dispatcher   │
│ topoSort,   │ │ op fusion,  │ │  routes subgraphs    │
│ wave fronts │ │ CSE, DCE    │ │  to backend impls    │
└──────┬──────┘ └─────┬──────┘ └────┬────────────────┘
       └──────────────┴─────────────┘
                      │
       ┌──────────────┼──────────────┐
       │              │              │
┌──────▼──────┐ ┌─────▼──────┐ ┌────▼──────┐
│    Metal    │ │    CPU /    │ │   CUDA /  │
│  (shaders)  │ │  Accelerate │ │    ANE    │
│  primary    │ │  + naive    │ │  (stubs)  │
└─────────────┘ └────────────┘ └───────────┘
```

---

## Core Types

### Device

Device is a value type (enum) attached to each tensor. Every tensor knows where it lives. Cross-device transfers are explicit ops in the graph, never silent side effects.

```
Device
  ├── cpu
  ├── metal(deviceIndex: Int)
  ├── cuda(deviceIndex: Int)     // future
  └── ane(deviceIndex: Int)      // future, see ANE section
```

### DType

```
DType
  ├── float32
  ├── float16
  ├── bfloat16
  ├── int32
  └── bool
```

### Tensor

`Tensor` is a lightweight handle to a graph node. It does not hold raw memory. The fields it exposes are derived from the underlying `GraphNode`:

- `shape: [Int]`
- `dtype: DType`
- `device: Device`
- `requiresGrad: Bool`
- `gradFn: GradFn?` — nil until autograd is implemented, wired now

Materialization is triggered by inspecting the value (e.g. converting to a Swift array). On Apple Silicon with shared memory, this requires no copy — it is a CPU pointer read into the `MTLBuffer`.

---

## Computation Graph (IR)

Every operation produces a `GraphNode`. Nodes form a directed acyclic graph (DAG) where edges represent data dependencies.

### GraphNode fields

| Field | Purpose |
|---|---|
| `id` | Unique identifier |
| `op` | The operation this node computes |
| `inputs` | Dependency edges (other nodes) |
| `shape` | Output shape |
| `dtype` | Output dtype |
| `device` | Target device |
| `storage` | Nil until realized; holds the backend buffer |
| `savedForBackward` | Reserved for autograd |
| `gradAccumulator` | Reserved for autograd |

### Op Vocabulary

The op set is intentionally small. Covering ~30–40 primitives handles the vast majority of inference workloads.

**Two tiers exist:**

1. **Primitive ops** — the scheduler understands these deeply and can fuse, reorder, or eliminate them
2. **Compound ops** — opaque to the scheduler, implemented as a single Metal shader, but required to register a decomposition for autograd later (e.g. `gridSample`, `scaledDotProductAttention`, `rotaryEmbedding`)

Primitive op categories:

- Creation: `constant`, `zeros`, `ones`, `randn`
- Shape: `reshape`, `permute`, `broadcast`, `slice`
- Elementwise: `add`, `mul`, `sub`, `div`, `relu`, `gelu`, `silu`, `exp`, `log`, `sqrt`
- Reduction: `sum`, `mean`, `max`
- Linear algebra: `matmul`, `conv2d`
- Normalization: `layerNorm`
- Movement: `toDevice(Device)` — explicit transfer, never implicit

Every op, including compound ops, must have its VJP (vector-Jacobian product) documented in comments even before autograd is implemented. This enforces that only differentiable ops are added, or that non-differentiable ops are consciously marked as such.

---

## Storage

### StorageBuffer Protocol

`StorageBuffer` is a **protocol**, not a concrete type. This is the critical portability boundary. The graph layer, scheduler, and dispatcher hold `any StorageBuffer` and never import Metal or any platform-specific framework.

Required surface:

- `shape: [Int]`
- `dtype: DType`
- `device: Device`
- Copy bytes out (for user inspection / CPU fallback)
- Copy bytes in (for tensor creation from Swift arrays)

### Backend Implementations

| Implementation | Backing storage | Platform |
|---|---|---|
| `MetalStorageBuffer` | `MTLBuffer` | Apple only |
| `CPUStorageBuffer` | `UnsafeMutableBufferPointer` / `malloc` | All platforms |
| `ANEStorageBuffer` | TBD (future AIKit API) | Apple only |

Metal is never imported outside the `Backends/Metal` SPM target. If any file outside that target imports Metal, it is an architecture violation. Swift's module system enforces this at compile time.

### Metal Buffer Modes (Apple Silicon)

| Mode | CPU access | GPU access | Notes |
|---|---|---|---|
| Shared | Yes | Yes | Default on Apple Silicon; no PCIe penalty |
| Private | No | Yes | Fastest for GPU-only intermediates |
| Managed | Explicit sync | Yes | Legacy; not recommended on Apple Silicon |

Shared memory is the sensible default on Apple Silicon due to unified memory architecture. Intermediate activation buffers use a **buffer pool** — fixed-size buffers are returned to a free list rather than deallocated, which is essential for inference loops running the same graph repeatedly.

---

## Device Dispatcher

The dispatcher holds a registry of `BackendExecutor` implementations, keyed by `Device`. It is the only component that knows which backends are available.

### BackendExecutor Protocol

Every backend implements:

- `device: Device`
- `execute(node:inputs:) -> StorageBuffer`
- `supports(op:) -> Bool` — used by the scheduler to make fusion decisions
- `allocate(shape:dtype:) -> StorageBuffer`
- `deallocate(_ buffer:)`
- `synchronize()` — async; critical for correctness across command buffer boundaries

### Fallback Chain

When a backend does not support an op, the dispatcher falls back:

```
Metal → CPU Accelerate → Naive Swift
```

This means every op must have a naive Swift implementation. The naive backend is also the correctness oracle for testing shader output.

### Device Transfer Rule

`toDevice()` is a **first-class op in the graph**. Transfers are never implicit side effects of assignment. This means:

- The scheduler can see and minimize transfers
- Transfers are differentiable (gradient flows across device boundaries)
- Profiling can identify exactly where cross-device copies occur
- The autograd engine never has to discover that a transfer happened

This is the single most common architectural mistake in homegrown ML libraries and is essentially free to get right from the start.

---

## Scheduler and Executor

### Topological Sort

The executor sorts the graph into **wave fronts** — sets of nodes whose inputs are all already resolved. Nodes within a wave are independent and can execute in parallel.

On Metal, a wave maps naturally to a batch of compute commands encoded into a single `MTLCommandBuffer`, dispatched together.

### Graph Passes (Stub)

The optimizer pass pipeline is present in the architecture but unpopulated initially:

- **Common subexpression elimination (CSE)** — deduplicate identical subgraphs
- **Dead code elimination (DCE)** — drop nodes whose output is never used
- **Op fusion** — merge elementwise chains into a single shader dispatch

These passes operate on the graph before any backend sees the work.

---

## Metal Backend (Primary)

The Metal backend implements `BackendExecutor` using raw Metal compute shaders — no MPSGraph. This is a deliberate educational choice: MPSGraph is a black box, while raw shaders expose the full GPU execution model.

### What this requires

- A `.metal` shader file per op
- Shaders compiled into a `MTLLibrary` at startup
- Per-execution: encode a `MTLComputeCommandEncoder`, bind `MTLBuffer` inputs and outputs, dispatch threadgroups
- A JIT-style cache keyed on op signature to avoid recompiling pipelines

### Hard problems encountered by design

- **Threadgroup sizing** — correct occupancy on Apple Silicon's GPU topology requires understanding the hardware's SIMD width and register file
- **Tiled matrix multiplication** — a naive matmul shader is approximately 10× slower than a tiled one; implementing tiling teaches cache hierarchy, threadgroup shared memory, and why BLAS exists
- **Numerical precision** — float16 vs float32 behavior differs subtly between GPU and CPU; the naive backend comparison catches these bugs
- **Buffer aliasing** — determining when two ops can share the same `MTLBuffer` without a copy is a real scheduler concern

---

## Build System: Swift Package Manager

CMake is not used initially. SPM is preferred because:

- `#if canImport(Metal)` evaluates to false on Linux automatically — no manual flags needed for the primary portability concern
- Platform conditions are first-class in SPM manifests
- Module boundaries are enforced by the compiler without additional build system configuration

### Target Structure

```
MyML/                     ← public API, Tensor, operator overloads
MyMLCore/                 ← IR, GraphNode, Op, Device, DType
MyMLScheduler/            ← toposort, wave decomposition, pass pipeline
MyMLDispatch/             ← BackendExecutor protocol, DeviceDispatcher
MyMLBackendMetal/         ← Metal shaders, MTLBuffer management (Apple only)
MyMLBackendCPU/           ← Accelerate/vDSP, naive Swift fallback (all platforms)
MyMLBackendCUDA/          ← stub, C interop layer (future)
MyMLBackendANE/           ← stub (future, see below)
```

`MyML` is the single public product target. All backend targets are internal dependencies. Users import only `MyML`.

When ready to ship a single binary, SPM's `@_exported import` allows all internal targets to be re-exported through one library product.

---

## ANE Backend (Future)

Apple Neural Engine access today requires going through CoreML, which is opaque. A future `AIKit` API (analogous to the Metal → MPS → MPSGraph maturation path) would expose direct ANE programming.

### What to do now

Add `case ane(deviceIndex: Int = 0)` to the `Device` enum. This single line means every tensor in the system can express ANE placement today. The dispatcher gracefully falls back to Metal when no ANE executor is registered.

The `ANEBackendExecutor` stub should implement `supports(op:)` with all ops returning false initially. The ANE hardware has known constraints that will shape which ops eventually return true:

- Strongly prefers quantized (fixed-point) arithmetic
- Limited dynamic shape support
- No arbitrary compute kernels
- Specific layout requirements (channel-last preferred)

Documenting these as capability flags now means implementation is filling in values, not redesigning the interface.

---

## Autograd Readiness

Autograd is not implemented. The following structural decisions ensure it is not an afterthought when it is.

| Decision | Why it matters for autograd |
|---|---|
| Lazy graph is the tape | No second data structure needed; the forward graph already encodes the computation history |
| `savedForBackward` on every node | Forward pass populates this; backward pass reads it |
| `gradFn` on every Tensor | Populated by autograd; nil during inference |
| `toDevice()` is a graph op | Gradient flows across device boundaries without special cases |
| Every op has a documented VJP | Forces the op vocabulary to only contain differentiable ops, or to consciously mark non-differentiable ones |
| Two-tier op vocabulary | Compound ops register a decomposition; the autograd engine uses it for the backward pass without needing to understand the fused forward shader |

---

## Testing Strategy

Every op must pass correctness tests against the naive pure-Swift CPU backend before any optimized implementation is considered correct. The naive backend:

- Has no GPU dependency
- Runs on Linux CI without hardware
- Produces float32 reference values against which Metal shader output is diffed
- Catches threadgroup boundary bugs, which are otherwise nearly invisible

Typical test structure: generate random inputs, run both backends, assert outputs are within a tolerance appropriate for the dtype.

---

## Summary of Key Architectural Decisions

| Decision | Alternative considered | Reason chosen |
|---|---|---|
| Lazy evaluation | Eager (PyTorch default) | Graph is free for autograd and fusion; MPSGraph is natively a graph compiler |
| StorageBuffer as protocol | Concrete struct with Metal fields | Enforces Linux portability at compile time |
| `toDevice()` as explicit graph op | Implicit transfer on assignment | Visible to scheduler, differentiable, profilable |
| Raw Metal shaders | MPSGraph | Educational transparency; full visibility into GPU execution |
| Two-tier op vocabulary | Flat op list | Compound ops need opaque forward shaders but transparent autograd decompositions |
| SPM over CMake | CMake with `MYML_HAS_METAL` | `canImport(Metal)` handles primary portability case automatically |
| ANE device enum value now | Add when API ships | Zero cost; reserves placement semantics across the entire type system |
