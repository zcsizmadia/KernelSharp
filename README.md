# KernelSharp ⚡

> **Write CUDA kernels in C#. Compile at build time. Ship as a NuGet package.**

KernelSharp is a .NET library that lets you write CUDA C/C++ kernels
**directly inside your C# source files** — no cmake, no separate `.cu` build system, no
Visual Studio or nvcc required, no runtime JIT overhead, no CUDA Runtime API boilerplate.
Annotate a `partial` method with `[GpuKernel]`, write the kernel inline with a raw string
literal, and by the time your project finishes building the kernel is **compiled, optionally
brotli-compressed, and embedded directly in your assembly**. Runtime dispatch happens through
the CUDA Driver API with zero managed allocations on the hot path.

Compilation uses **NVRTC** (NVIDIA Runtime Compilation) — an in-process library that ships
with the CUDA Toolkit. No subprocess, no `cl.exe`, no nvcc path detection needed.

---

## Why KernelSharp?

| Feature | KernelSharp | Typical CUDA .NET wrapper |
|---|---|---|
| Kernel source lives next to its C# caller | ✅ inline raw string | ❌ separate .cu / .ptx file |
| Build-time compilation, no subprocess | ✅ NVRTC in-process | ❌ manual CMake / MSBuild targets |
| No Visual Studio / cl.exe dependency | ✅ NVRTC only | ❌ VS + MSVC required on Windows |
| Runtime compilation to native GPU arch | ✅ `Compilation = Runtime` mode | ❌ N/A |
| Strongly-typed device buffers | ✅ `CudaBuffer<T>` | ❌ raw `IntPtr` |
| NuGet-installable, no CUDA SDK at runtime | ✅ PTX embedded in DLL | ❌ SDK / driver headers required |
| Parallel kernel compilation | ✅ all cores | ❌ N/A |
| Single NuGet package | ✅ runtime + build task in one | ❌ separate packages |

---

## Quick Start

### 1 — Add the NuGet package

```xml
<PackageReference Include="KernelSharp" Version="1.0.0" />
```

A single package provides both the runtime (`CudaBuffer<T>`, `CudaContext`, …) and the
MSBuild task that invokes NVRTC at build time.

### 2 — Write your first kernel

```csharp
using KernelSharp;

public partial class MyKernels
{
    [GpuKernel("""
        extern "C" __global__ void AddVectors(
            const float* __restrict__ a,
            const float* __restrict__ b,
            float*       __restrict__ c,
            int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) c[i] = a[i] + b[i];
        }
        """)]
    public partial void AddVectors(CudaBuffer<float> a, CudaBuffer<float> b, CudaBuffer<float> c);
}
```

### 3 — Call it like any C# method

```csharp
using var ctx = CudaContext.Initialize();  // initialises the CUDA Driver API

const int N = 1 << 20;                    // 1M elements

using var dA = CudaBuffer<float>.Allocate(N);
using var dB = CudaBuffer<float>.Allocate(N);
using var dC = CudaBuffer<float>.Allocate(N);

float[] hA = Enumerable.Range(0, N).Select(i => (float)i).ToArray();
float[] hB = Enumerable.Range(0, N).Select(i => (float)i * 2f).ToArray();

dA.CopyFromHost(hA);
dB.CopyFromHost(hB);

var kernels = new MyKernels();
kernels.AddVectors(dA, dB, dC);   // generated launch wrapper

float[] result = new float[N];
dC.CopyToHost(result);
Console.WriteLine(result[0]);   // 0.0 + 0.0 = 0.0 ✓
```

> **Multithreaded / parallel usage** — CUDA Driver API contexts are thread-local.
> If you share a single `CudaContext` across threads (e.g. in a test suite with parallel
> test execution), call `ctx.MakeCurrent()` at the start of each thread before issuing
> any GPU operations. `CudaContext.Initialize()` only makes the context current on the
> thread that called it.

No `cuModuleLoad`, no `cuLaunchKernel`, no kernel argument marshalling — **the MSBuild
task writes all of that code for you**.

---

## The `[GpuKernel]` Attribute

### Inline source (recommended)

Use a C# 11 raw string literal to embed CUDA C/C++ directly. No escaping needed:

```csharp
[GpuKernel("""
    extern "C" __global__ void ReLU(const float* x, float* y, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) y[i] = fmaxf(x[i], 0.f);
    }
    """)]
public partial void ReLU(CudaBuffer<float> x, CudaBuffer<float> y);
```

### External `.cu` file

Point to a file on disk relative to the declaring C# source file:

```csharp
[GpuKernel(SourceFile = "Kernels/flash_attn.cu")]
public partial void FlashAttn(
    CudaBuffer<float> q, CudaBuffer<float> k,
    CudaBuffer<float> v, CudaBuffer<float> o,
    int seqLen, int headDim);
```

### Per-kernel overrides

```csharp
[GpuKernel("""...""",
    Arch           = "compute_89",          // single arch for build-time compilation
    ExtraFlags     = "-lineinfo",           // NVRTC options
    IncludePath    = "vendor/cutlass/include",
    Compression    = "none",               // "brotli" (default), "gzip", "zlib", "deflate", "none"
    Compilation    = KernelCompilation.Runtime,  // per-kernel mode override
    ThreadsPerBlock = 128,                 // override default 256-thread blocks
    BlocksPerGrid  = 4)]                  // or fix the block count entirely
public partial void MyKernel(CudaBuffer<float> a, CudaBuffer<float> b);
```

The `ThreadsPerBlock` / `BlocksPerGrid` properties control the `cuLaunchKernel` grid:

| `ThreadsPerBlock` | `BlocksPerGrid` | Generated launch |
|---|---|---|
| 0 (default) | 0 (default) | `threads=256, blocks=ceil(n/256)` |
| T | 0 | `threads=T, blocks=ceil(n/T)` |
| 0 | B | `threads=256, blocks=B` |
| T | B | `threads=T, blocks=B` (fully fixed — e.g. single-block scans) |

### Stub during development

```csharp
[GpuKernel("""...""", NotImplemented = true)]
public partial void ExperimentalKernel(CudaBuffer<float> x);
// → throws NotImplementedException at runtime; NVRTC is never invoked at build time
```

---

## Strongly-Typed Device Buffers

`CudaBuffer<T>` is a typed wrapper around a CUDA device pointer. The element type is
fixed at declaration time so the compiler catches host/device type mismatches early:

```csharp
// Allocation — element count, not byte count
using var weights = CudaBuffer<float>.Allocate(hiddenDim);
using var tokens  = CudaBuffer<int>.Allocate(seqLen);
using var packed  = CudaBuffer<byte>.Allocate(quantBytes);

// Host ↔ Device transfers accept arrays or Span<T>
weights.CopyFromHost(floatArray);
weights.CopyFromHost(spanOfFloat);
weights.CopyToHost(destination);

// Introspect without touching the GPU
long byteSize = weights.ByteSize;   // elementCount * sizeof(float)
int  count    = weights.Length;
IntPtr ptr    = weights.DevicePointer;
```

Non-float example — int4 dequantisation kernel:

```csharp
[GpuKernel("""
    extern "C" __global__ void DequantInt4(
        const uint8_t* packed, const float* scales, float* output, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        uint8_t b = packed[i >> 1];
        float   v = (i & 1) ? (b >> 4) : (b & 0xF);
        output[i] = (v - 8.f) * scales[i >> 128];
    }
    """)]
public partial void DequantInt4(
    CudaBuffer<byte>  packed,
    CudaBuffer<float> scales,
    CudaBuffer<float> output,
    int n);
```

---

## How Build-Time Compilation Works

```
dotnet build
    │
    ├─ Roslyn compiles your C# code (including [GpuKernel] declarations)
    │
    └─ KernelSharp MSBuild task runs (BeforeTargets="CoreCompile")
           │
           ├─ Scans all .cs files for [GpuKernel] on partial methods
           ├─ Extracts inline source or reads the referenced .cu file
           ├─ Classifies each parameter:
           │     CudaBuffer<T>  → Buffer  → extract .DevicePointer
           │     int/float/...  → Scalar  → pass value directly
           ├─ Calls NVRTC in-process (parallel, all CPU cores by default)
           │     one NVRTC program handle per [GpuKernel] method
           ├─ Collects resulting PTX bytes
           ├─ Optionally compresses the PTX (brotli by default)
           └─ Emits  MyClass.MyMethod.g.cs  containing:
                  • static readonly byte[] _ptx_encoded = { … };
                  • static byte[]  _ptx = KernelBlobHelper.Decode(…);
                  • static IntPtr _module, _func;
                  • public partial void MyMethod(…) { … cuLaunchKernel(…) }

    └─ Roslyn compiles the generated .g.cs files alongside your code
           → single assembly, zero external resources
```

### Incremental builds

The MSBuild task uses timestamp-based incremental compilation. If a kernel's source file
hasn't changed since the last build, NVRTC is not re-invoked. Cold builds (all kernels
new) compile in parallel; warm builds (no changes) add essentially zero overhead.

---

## Checking In Generated Files (CI without NVRTC)

By default the generated `.g.cs` launcher files are written to `$(IntermediateOutputPath)`
and are not committed to source control. If you want build machines that don't have CUDA
installed to be able to compile your project, set `KernelSharpGeneratedOutputPath` to a
committed folder:

```xml
<PropertyGroup>
  <!-- Commit generated launchers so CI machines without NVRTC can compile -->
  <KernelSharpGeneratedOutputPath>Generated\</KernelSharpGeneratedOutputPath>
</PropertyGroup>
```

When this property is set:
- Generated `.g.cs` files are written to (and read from) that folder instead of `obj/`.
- NVRTC is still skipped when the generated file is **newer** than the source `.cs` file.
- Machines without the CUDA Toolkit can compile using the checked-in launchers.

---

## NVRTC Library Discovery

KernelSharp locates the NVRTC library automatically. Search order:

1. `KERNELSHARP_CUDA_PATH` environment variable
2. `CUDA_PATH` environment variable
3. `CUDA_TOOLKIT_ROOT_DIR` environment variable
4. `PATH` entries
5. **Windows**: `%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvrtc64_*.dll` (newest first)
6. **Linux**: `/usr/local/cuda/lib64/libnvrtc.so`, then `/usr/lib/x86_64-linux-gnu/libnvrtc.so.*`

---

## MSBuild Configuration

All settings have sensible defaults. Override only what you need:

```xml
<!-- In your .csproj or Directory.Build.props -->
<PropertyGroup>
  <!-- Path to an extra CUDA include directory -->
  <KernelSharpIncludePath>C:\libs\cuda\include</KernelSharpIncludePath>

  <!-- Minimum PTX virtual architecture for build-time compilation.
       Default: compute_75 (Turing, 2018+). The driver JIT-compiles PTX to native SASS.
       Accepts: "compute_75", "sm_89", bare "80", etc. -->
  <KernelSharpMinArch>compute_80</KernelSharpMinArch>

  <!-- Extra NVRTC options for every kernel in this project (space-separated).
       Per-kernel overrides go in [GpuKernel(ExtraFlags = "…")]. -->
  <KernelSharpExtraOptions>-lineinfo</KernelSharpExtraOptions>

  <!-- Default compilation mode: BuildTime (default) or Runtime -->
  <KernelSharpCompilation>BuildTime</KernelSharpCompilation>

  <!-- Max parallel NVRTC compilations; empty = all CPU cores -->
  <KernelSharpMaxParallelism>4</KernelSharpMaxParallelism>

  <!-- PTX embedding compression: brotli (default, best ratio), gzip, zlib, deflate, or none -->
  <KernelSharpPtxCompression>brotli</KernelSharpPtxCompression>

  <!-- Optional: write generated .g.cs files to a committed folder (see above) -->
  <KernelSharpGeneratedOutputPath>Generated\</KernelSharpGeneratedOutputPath>
</PropertyGroup>
```

When installed as a NuGet package, `build/KernelSharp.props` is auto-imported and sets
all these defaults — no manual setup required.

---

## Build Diagnostics

| Code | Severity | Meaning |
|---|---|---|
| `KERNELSHARP001` | Warning | NVRTC library not found — kernel compilation skipped, kernels will fail at runtime |
| `KERNELSHARP002` | Error | NVRTC reported a compilation error — build fails with the NVRTC error log |
| `KERNELSHARP003` | Warning | Mismatch between the `__global__` function name or parameter count in the CUDA source and the C# method declaration. The actual CUDA function name is still used for `cuModuleGetFunction`; this warning just flags the inconsistency so it can be fixed before it causes a runtime error. |

---

## Real-World Kernel Examples

### Inclusive prefix scan (single-block Hillis-Steele)

```csharp
[GpuKernel("""
    extern "C" __global__ void PrefixScan(
        const float* __restrict__ x,
        float*       __restrict__ y,
        int n)
    {
        // Single-block, shared-memory Hillis-Steele inclusive scan (≤256 elements).
        __shared__ float smem[256];
        int tid = threadIdx.x;
        smem[tid] = (tid < n) ? x[tid] : 0.f;
        __syncthreads();
        for (int d = 1; d < blockDim.x; d <<= 1) {
            float v = (tid >= d) ? smem[tid - d] : 0.f;
            __syncthreads();
            smem[tid] += v;
            __syncthreads();
        }
        if (tid < n) y[tid] = smem[tid];
    }
    """, ThreadsPerBlock = 256, BlocksPerGrid = 1)]   // ← fixed single-block launch
public partial void PrefixScan(CudaBuffer<float> x, CudaBuffer<float> y);
```

`ThreadsPerBlock = 256, BlocksPerGrid = 1` tells the generator to emit a literal
`_threads=256; _blocks=1;` rather than the default auto-compute. This is required for
any single-block cooperative algorithm (scans, reductions that use `__syncthreads`
across the whole grid, etc.).

### Attention scores (transformer self-attention)

```csharp
[GpuKernel("""
    extern "C" __global__ void AttnScores(
        const float* __restrict__ q,
        const float* __restrict__ k,
        float*       __restrict__ scores,
        int seqLen, int headDim)
    {
        int row = blockIdx.x, col = threadIdx.x;
        if (col >= seqLen) return;
        float dot = 0.f;
        for (int d = 0; d < headDim; d++)
            dot += q[row * headDim + d] * k[col * headDim + d];
        scores[row * seqLen + col] = dot * rsqrtf((float)headDim);
    }
    """)]
public partial void AttnScores(
    CudaBuffer<float> q,
    CudaBuffer<float> k,
    CudaBuffer<float> scores,
    int seqLen, int headDim);
```

### RMS Normalisation (LLaMA / Mistral)

```csharp
[GpuKernel("""
    extern "C" __global__ void RMSNorm(
        const float* x, const float* weight, float* y, int n, float eps)
    {
        float sum = 0.f;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            sum += x[i] * x[i];
        __shared__ float shared;
        if (threadIdx.x == 0) shared = rsqrtf(sum / n + eps);
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            y[i] = x[i] * shared * weight[i];
    }
    """)]
public partial void RMSNorm(
    CudaBuffer<float> x,
    CudaBuffer<float> weight,
    CudaBuffer<float> y,
    int n, float eps);
```

### Embedding lookup (token → hidden state)

```csharp
[GpuKernel("""
    extern "C" __global__ void EmbedLookup(
        const int* tokenIds, const float* table, float* output,
        int hiddenDim)
    {
        int tok = blockIdx.x, d = threadIdx.x;
        if (d < hiddenDim)
            output[tok * hiddenDim + d] = table[tokenIds[tok] * hiddenDim + d];
    }
    """)]
public partial void EmbedLookup(
    CudaBuffer<int>   tokenIds,
    CudaBuffer<float> table,
    CudaBuffer<float> output,
    int hiddenDim);
```

---

## Supported Platforms & Requirements

| | Windows | Linux |
|---|---|---|
| NVRTC (build-time) | `nvrtc64_*.dll` (CUDA Toolkit 11+) | `libnvrtc.so` (CUDA Toolkit 11+) |
| CUDA Driver (runtime) | `nvcuda.dll` (display driver) | `libcuda.so` (display driver) |
| .NET target | net8.0, net9.0, net10.0 | net8.0, net9.0, net10.0 |
| Visual Studio / cl.exe | ❌ not required | ❌ not required |

The **CUDA Runtime API is not required** at runtime. KernelSharp uses only the
CUDA Driver API (`nvcuda.dll` / `libcuda.so`), which ships with the display driver —
no CUDA SDK installation needed on end-user machines.

---

## Package

| Package | Purpose |
|---|---|
| `KernelSharp` | Runtime + build task: `CudaBuffer<T>`, `CudaContext`, `CudaStream`, Driver API P/Invokes, NVRTC bindings, and the MSBuild task that compiles CUDA kernels at build time |

A single package covers everything. No separate generator package is needed.

> **Implementation note:** inside the package, the MSBuild task lives in
> `build/KernelSharp.Build.dll` (a separate assembly from the runtime `KernelSharp.dll`).
> This prevents the MSBuild host from locking `KernelSharp.dll` during builds,
> allowing incremental rebuilds of the library itself without DLL-lock errors.
> Both assemblies share NVRTC bindings via the `KernelSharp.Nvrtc` shared project.

---

## Building from Source

### Prerequisites

| Tool | Notes |
|---|---|
| [.NET 10 SDK](https://dotnet.microsoft.com/download) | Required |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (11.0+) | Required to compile `.cu` kernels at build time; `nvrtc64_*.dll` (Windows) or `libnvrtc.so` (Linux) must be discoverable |

No Visual Studio or MSVC required on any platform.

### Repository layout

```
KernelSharp.Build/        ← MSBuild task (CompileCudaKernelsTask)
                            packed into build/KernelSharp.Build.dll in the NuGet package
KernelSharp.Nvrtc/        ← Shared project: NVRTC P/Invoke bindings
                            compiled into both KernelSharp.dll and KernelSharp.Build.dll
KernelSharp/              ← Runtime library (CudaBuffer<T>, CudaContext, …)
  build/
    KernelSharp.props     ← auto-imported MSBuild properties
    KernelSharp.targets   ← UsingTask + KernelSharp_CompileCudaKernels target
KernelSharp.Samples/      ← example kernels
KernelSharp.Tests/        ← unit + integration tests (TUnit)
```

### Build

```sh
dotnet build
```

Build order is `KernelSharp.Build` → `KernelSharp` → `KernelSharp.Samples` /
`KernelSharp.Tests`. MSBuild respects the `ProjectReference` dependency chain, so
`KernelSharp.Build.dll` is always ready before the task is needed.

### Run tests

```sh
dotnet test
```

Tests that require a physical GPU are skipped automatically when no CUDA-capable
device is detected. The code-generation tests (`SourceGeneratorTests`) run without a
GPU.

### Pack

```sh
dotnet pack KernelSharp/KernelSharp.csproj -c Release
```

The produced `.nupkg` contains:
- `lib/net10.0/KernelSharp.dll` — runtime assembly
- `build/KernelSharp.targets` — MSBuild target
- `build/KernelSharp.props` — MSBuild properties
- `build/KernelSharp.Build.dll` — MSBuild task assembly (loaded by MSBuild, never by end-user code)

---

## License

MIT — see [LICENSE](LICENSE).

