# KernelSharp ⚡

> **Write CUDA kernels in C#. Compile at build time. Ship as a NuGet package.**

KernelSharp is a .NET library that lets you write CUDA C/C++ kernels
**directly inside your C# source files** — no cmake, no separate `.cu` build system, no runtime
JIT overhead, no CUDA Runtime API boilerplate. Annotate a `partial` method with
`[GpuKernel]`, write the kernel inline with a raw string literal, and by the time your
project finishes building the kernel is **compiled, multi-arch, gzip-optionally-compressed,
and embedded directly in your assembly**. Runtime dispatch happens through the CUDA Driver
API with zero managed allocations on the hot path.

---

## Why KernelSharp?

| Feature | KernelSharp | Typical CUDA .NET wrapper |
|---|---|---|
| Kernel source lives next to its C# caller | ✅ inline raw string | ❌ separate .cu / .ptx file |
| Build-time nvcc compilation | ✅ MSBuild task, parallel | ❌ manual CMake / MSBuild targets |
| Multi-arch fatbin (Ampere, Ada, Hopper …) | ✅ automatic | ❌ per-arch manual flags |
| Strongly-typed device buffers | ✅ `CudaBuffer<T>` | ❌ raw `IntPtr` |
| Zero-config compiler auto-detection | ✅ nvcc + MSVC auto-discovered | ❌ path config required |
| NuGet-installable, no CUDA SDK at runtime | ✅ fatbin embedded in DLL | ❌ SDK / driver headers required |
| Parallel kernel compilation | ✅ all cores | ❌ N/A |
| Single NuGet package | ✅ runtime + build task in one | ❌ separate packages |

---

## Quick Start

### 1 — Add the NuGet package

```xml
<PackageReference Include="KernelSharp" Version="1.0.0" />
```

A single package provides both the runtime (`CudaBuffer<T>`, `CudaContext`, …) and the
MSBuild task that invokes nvcc at build time.

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
using var ctx    = new CudaContext();          // initialises the CUDA Driver API
using var stream = new CudaStream();

const int N = 1 << 20;                        // 1M elements

using var dA = CudaBuffer<float>.Allocate(N);
using var dB = CudaBuffer<float>.Allocate(N);
using var dC = CudaBuffer<float>.Allocate(N);

float[] hA = Enumerable.Range(0, N).Select(i => (float)i).ToArray();
float[] hB = Enumerable.Range(0, N).Select(i => (float)i * 2f).ToArray();

dA.CopyFromHost(hA);
dB.CopyFromHost(hB);

var kernels = new MyKernels();
kernels.AddVectors(dA, dB, dC, stream: stream);   // generated launch wrapper

float[] result = new float[N];
dC.CopyToHost(result);
Console.WriteLine(result[0]);   // 0.0 + 0.0 = 0.0 ✓
```

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
    Arch        = "compute_89",          // single arch — faster debug builds
    ExtraFlags  = "-lineinfo -G",        // add device debug info
    IncludePath = "vendor/cutlass/include")]
public partial void MyKernel(CudaBuffer<float> a, CudaBuffer<float> b);
```

### Stub during development

```csharp
[GpuKernel("""...""", NotImplemented = true)]
public partial void ExperimentalKernel(CudaBuffer<float> x);
// → throws NotImplementedException at runtime; nvcc is never invoked at build time
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
           ├─ Spawns nvcc processes in parallel (all CPU cores by default)
           │     one process per [GpuKernel] method
           ├─ Collects resulting fatbin bytes
           ├─ Optionally gzip-compresses the fatbin
           └─ Emits  MyClass.MyMethod.g.cs  containing:
                  • static readonly byte[] _fatbin = { … };
                  • static IntPtr _module, _func;
                  • public partial void MyMethod(…) { … cuLaunchKernel(…) }

    └─ Roslyn compiles the generated .g.cs files alongside your code
           → single assembly, zero external resources
```

The generated file includes a build-metadata comment showing the exact nvcc command line
that produced the fatbin, compiler versions, and the date — making the build fully
reproducible and auditable.

### Incremental builds

The MSBuild task uses timestamp-based incremental compilation. If a kernel's source file
hasn't changed since the last build, nvcc is not re-invoked. Cold builds (all kernels
new) compile in parallel; warm builds (no changes) add essentially zero overhead.

---

## Checking In Generated Files (CI without nvcc)

By default the generated `.g.cs` launcher files are written to `$(IntermediateOutputPath)`
and are not committed to source control. If you want build machines that don't have CUDA
installed to be able to compile your project, set `KernelSharpGeneratedOutputPath` to a
committed folder:

```xml
<PropertyGroup>
  <!-- Commit generated launchers so CI machines without nvcc can compile -->
  <KernelSharpGeneratedOutputPath>Generated\</KernelSharpGeneratedOutputPath>
</PropertyGroup>
```

When this property is set:
- Generated `.g.cs` files are written to (and read from) that folder instead of `obj/`.
- nvcc is still skipped when the generated file is **newer** than the source `.cs` file.
- Machines without nvcc can compile using the checked-in launchers.

---

## Compiler Auto-Detection

KernelSharp finds your compilers automatically — no path configuration required for most
setups. Configuration properties are available for unusual installations.

### nvcc detection order

1. `CUDA_PATH` environment variable → `$CUDA_PATH/bin/nvcc`
2. `CUDA_TOOLKIT_ROOT_DIR` environment variable → `$CUDA_TOOLKIT_ROOT_DIR/bin/nvcc`
3. `PATH` — each entry is checked for `nvcc` / `nvcc.exe`
4. **Windows** — `%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvcc.exe`  
   (all installed versions, newest first)
5. **Linux** — `/usr/local/cuda/bin/nvcc`, then `/usr/bin/nvcc`

### MSVC `cl.exe` detection order (Windows only)

nvcc requires a compatible host C++ compiler on Windows. KernelSharp finds it without
needing Visual Studio to be open or any environment pre-activation:

1. `KernelSharpMsvcClPath` MSBuild property — explicit full path or directory
2. `VCToolsInstallDir` environment variable (set by `vcvarsall.bat`)
3. **vswhere** — `%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe`  
   Queries the latest pre-release or stable VS installation, enumerates MSVC toolchain
   versions inside it (newest first)
4. **Directory scan** — walks `%ProgramFiles%\Microsoft Visual Studio\` and
   `%ProgramFiles(x86)%\Microsoft Visual Studio\`, year directories newest-first,
   edition directories newest-first (Enterprise → Preview → Community …),
   MSVC toolchain versions newest-first
5. `PATH` — last resort, checks each entry for `cl.exe`

On Linux, GCC is picked up by nvcc automatically; no host compiler detection is needed.

---

## MSBuild Configuration

All settings have sensible defaults. Override only what you need:

```xml
<!-- In your .csproj or Directory.Build.props -->
<PropertyGroup>
  <!-- Path to the CUDA include root (if not in CUDA_PATH) -->
  <KernelSharpIncludePath>C:\libs\cuda\include</KernelSharpIncludePath>

  <!-- C++ standard: c++14, c++17, c++20 (default: c++20) -->
  <KernelSharpNvccStd>c++20</KernelSharpNvccStd>

  <!-- Explicit CCCL (Thrust / libcudacxx / CUB) root directory.
       Auto-detected from CUDA install (bundled since CUDA 12.4) when empty -->
  <KernelSharpCCCLPath></KernelSharpCCCLPath>

  <!-- Extra nvcc flags for every kernel in this project -->
  <KernelSharpNvccExtraFlags>-lineinfo</KernelSharpNvccExtraFlags>

  <!-- Explicit cl.exe path; auto-detected via vswhere when empty (Windows only) -->
  <KernelSharpMsvcClPath></KernelSharpMsvcClPath>

  <!-- Comma-separated target architectures (default: compute_80,compute_89,compute_90)
       Use a single arch for faster debug builds -->
  <KernelSharpTargetArchs>compute_80,compute_89,compute_90</KernelSharpTargetArchs>

  <!-- Max parallel nvcc processes; empty = all CPU cores -->
  <KernelSharpMaxParallelism>4</KernelSharpMaxParallelism>

  <!-- Fatbin embedding: gzip (default, ~50% smaller) or none (raw bytes) -->
  <KernelSharpFatbinCompression>gzip</KernelSharpFatbinCompression>

  <!-- Optional: write generated .g.cs files to a committed folder (see above) -->
  <KernelSharpGeneratedOutputPath>Generated\</KernelSharpGeneratedOutputPath>
</PropertyGroup>
```

When installed as a NuGet package, `build/KernelSharp.props` is auto-imported and sets
all these defaults — no manual setup required.

---

## Real-World Kernel Examples

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
| Host compiler | MSVC (VS 2019+, auto-detected) | GCC (picked up by nvcc) |
| nvcc version | CUDA 11.0+ | CUDA 11.0+ |
| .NET target | net8.0, net9.0, net10.0 | net8.0, net9.0, net10.0 |
| GPU architectures | sm_70 and newer | sm_70 and newer |

The **CUDA Runtime API is not required** at runtime. KernelSharp uses only the
CUDA Driver API (`nvcuda.dll` / `libcuda.so`), which ships with the display driver —
no CUDA SDK installation needed on end-user machines.

---

## Package

| Package | Purpose |
|---|---|
| `KernelSharp` | Runtime + build task: `CudaBuffer<T>`, `CudaContext`, `CudaStream`, Driver API P/Invokes, and the MSBuild task that compiles CUDA kernels at build time |

A single package covers everything. No separate generator package is needed.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Why KernelSharp?

| Feature | KernelSharp | Typical CUDA .NET wrapper |
|---|---|---|
| Kernel source lives next to its C# caller | ✅ inline raw string | ❌ separate .cu / .ptx file |
| Build-time nvcc compilation | ✅ Roslyn incremental generator | ❌ manual CMake / MSBuild targets |
| Multi-arch fatbin (Ampere, Ada, Hopper …) | ✅ automatic | ❌ per-arch manual flags |
| Strongly-typed device buffers | ✅ `CudaBuffer<T>` | ❌ raw `IntPtr` |
| Zero-config compiler auto-detection | ✅ nvcc + MSVC auto-discovered | ❌ path config required |
| NuGet-installable, no CUDA SDK at runtime | ✅ fatbin embedded in DLL | ❌ SDK / driver headers required |
| Parallel kernel compilation | ✅ all cores | ❌ N/A |

---

## Quick Start

### 1 — Add the NuGet packages

```xml
<PackageReference Include="KernelSharp" Version="1.0.0" />
<!-- KernelSharp.SourceGenerator is a build-time-only dependency pulled in transitively -->
```

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
using var ctx    = new CudaContext();          // initialises the CUDA Driver API
using var stream = new CudaStream();

const int N = 1 << 20;                        // 1M elements

using var dA = CudaBuffer<float>.Allocate(N);
using var dB = CudaBuffer<float>.Allocate(N);
using var dC = CudaBuffer<float>.Allocate(N);

float[] hA = Enumerable.Range(0, N).Select(i => (float)i).ToArray();
float[] hB = Enumerable.Range(0, N).Select(i => (float)i * 2f).ToArray();

dA.CopyFromHost(hA);
dB.CopyFromHost(hB);

var kernels = new MyKernels();
kernels.AddVectors(dA, dB, dC, stream: stream);   // generated launch wrapper

float[] result = new float[N];
dC.CopyToHost(result);
Console.WriteLine(result[0]);   // 0.0 + 0.0 = 0.0 ✓
```

No `cuModuleLoad`, no `cuLaunchKernel`, no kernel argument marshalling — **the source
generator writes all of that code for you**.

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
    Arch       = "compute_89",          // single arch — faster debug builds
    ExtraFlags = "-lineinfo -G",        // add device debug info
    IncludePath = "vendor/cutlass/include")]
public partial void MyKernel(CudaBuffer<float> a, CudaBuffer<float> b);
```

### Stub during development

```csharp
[GpuKernel("""...""", NotImplemented = true)]
public partial void ExperimentalKernel(CudaBuffer<float> x);
// → throws NotImplementedException at runtime; nvcc is never invoked at build time
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
    ├─ Roslyn compiles your C# code
    │      │
    │      └─ KernelSharp.SourceGenerator (IIncrementalGenerator)
    │              │
    │              ├─ Scans for [GpuKernel] on partial methods
    │              ├─ Extracts inline source or reads .cu file
    │              ├─ Classifies each parameter:
    │              │     CudaBuffer<T>  → Buffer  → extract .DevicePointer
    │              │     int/float/...  → Scalar  → pass value directly
    │              ├─ Spawns nvcc processes in parallel (all CPU cores)
    │              │     one process per [GpuKernel] method
    │              ├─ Collects resulting fatbin bytes
    │              ├─ Optionally gzip-compresses the fatbin
    │              └─ Emits  MyClass.MyMethod.g.cs  containing:
    │                    • static readonly byte[] _fatbin = { … };
    │                    • static IntPtr _module, _func;
    │                    • public partial void MyMethod(…) { … cuLaunchKernel(…) }
    │
    └─ Roslyn compiles the generated .g.cs files alongside your code
           → single assembly, zero external resources
```

The generated file includes a build-metadata comment showing the exact nvcc command line
that produced the fatbin, compiler versions, and the date — making the build fully
reproducible and auditable.

### Incremental builds

The Roslyn incremental generator tracks method signatures and source content. If neither
changed, the generator does **not** re-run nvcc. Cold builds (all kernels new) compile
in parallel; warm builds (no changes) add essentially zero overhead.

---

## Compiler Auto-Detection

KernelSharp finds your compilers automatically — no path configuration required for most
setups. Configuration properties are available for unusual installations.

### nvcc detection order

1. `CUDA_PATH` environment variable → `$CUDA_PATH/bin/nvcc`
2. `CUDA_TOOLKIT_ROOT_DIR` environment variable → `$CUDA_TOOLKIT_ROOT_DIR/bin/nvcc`
3. `PATH` — each entry is checked for `nvcc` / `nvcc.exe`
4. **Windows** — `%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvcc.exe`  
   (all installed versions, newest first)
5. **Linux** — `/usr/local/cuda/bin/nvcc`, then `/usr/bin/nvcc`

### MSVC `cl.exe` detection order (Windows only)

nvcc requires a compatible host C++ compiler on Windows. KernelSharp finds it without
needing Visual Studio to be open or any environment pre-activation:

1. `KernelSharpMsvcClPath` MSBuild property — explicit full path or directory
2. `VCToolsInstallDir` environment variable (set by `vcvarsall.bat`)
3. **vswhere** — `%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe`  
   Queries the latest pre-release or stable VS installation, enumerates MSVC toolchain
   versions inside it (newest first)
4. **Directory scan** — walks `%ProgramFiles%\Microsoft Visual Studio\` and
   `%ProgramFiles(x86)%\Microsoft Visual Studio\`, year directories newest-first,
   edition directories newest-first (Enterprise → Preview → Community …),
   MSVC toolchain versions newest-first
5. `PATH` — last resort, checks each entry for `cl.exe`

On Linux, GCC is picked up by nvcc automatically; no host compiler detection is needed.

---

## MSBuild Configuration

All settings have sensible defaults. Override only what you need:

```xml
<!-- In your .csproj or Directory.Build.props -->
<PropertyGroup>
  <!-- Path to the CUDA include root (if not in CUDA_PATH) -->
  <KernelSharpIncludePath>C:\libs\cuda\include</KernelSharpIncludePath>

  <!-- C++ standard: c++14, c++17, c++20 (default: c++20) -->
  <KernelSharpNvccStd>c++20</KernelSharpNvccStd>

  <!-- Explicit path to CCCL (Thrust / libcudacxx / CUB) root
       Auto-detected from CUDA install when empty -->
  <KernelSharpCCCLPath></KernelSharpCCCLPath>

  <!-- Extra nvcc flags for every kernel in this project -->
  <KernelSharpNvccExtraFlags>-lineinfo</KernelSharpNvccExtraFlags>

  <!-- Explicit cl.exe path; auto-detected via vswhere when empty (Windows) -->
  <KernelSharpMsvcClPath></KernelSharpMsvcClPath>

  <!-- Comma-separated target architectures (default: compute_80,compute_89,compute_90)
       Use a single arch for faster debug builds: compute_89 -->
  <KernelSharpTargetArchs>compute_80,compute_89,compute_90</KernelSharpTargetArchs>

  <!-- Max parallel nvcc processes; empty = all CPU cores -->
  <KernelSharpMaxParallelism>4</KernelSharpMaxParallelism>

  <!-- Fatbin embedding: none (default) or gzip (~50% smaller source) -->
  <KernelSharpFatbinCompression>none</KernelSharpFatbinCompression>
</PropertyGroup>
```

When installed as a NuGet package, `build/KernelSharp.SourceGenerator.props` is
auto-imported and sets all these defaults — no manual setup required.

---

## Real-World Kernel Examples

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
| Host compiler | MSVC (VS 2019+, auto-detected) | GCC (picked up by nvcc) |
| nvcc version | CUDA 11.0+ | CUDA 11.0+ |
| .NET target | net8.0, net9.0, net10.0 | net8.0, net9.0, net10.0 |
| GPU architectures | sm_70 and newer | sm_70 and newer |

The **CUDA Runtime API is not required** at runtime. KernelSharp uses only the
CUDA Driver API (`nvcuda.dll` / `libcuda.so`), which ships with the display driver —
no CUDA SDK installation needed on end-user machines.

---

## Packages

| Package | Purpose |
|---|---|
| `KernelSharp` | Runtime: `CudaBuffer<T>`, `CudaContext`, `CudaStream`, Driver API P/Invokes |
| `KernelSharp.SourceGenerator` | Build-time: Roslyn generator + MSBuild `.props` auto-import |

Add only `KernelSharp` to your project — it pulls in the source generator as a
build-time analyzer automatically.

---

## License

MIT — see [LICENSE](LICENSE).
