namespace KernelSharp;

/// <summary>
/// Marks a <c>partial</c> method whose CUDA C/C++ implementation is provided inline
/// or via an external file.  The KernelSharp MSBuild task compiles the CUDA source to
/// PTX using NVRTC, embeds the result in the assembly, and generates the method body
/// that loads and dispatches the kernel via the CUDA Driver API at runtime.
///
/// Usage – inline source (C# 11 raw string literal recommended):
/// <code>
///   [GpuKernel("""
///       extern "C" __global__ void AddVectors(
///           const float* a, const float* b, float* c, int n)
///       { int i = blockIdx.x*blockDim.x+threadIdx.x; if(i&lt;n) c[i]=a[i]+b[i]; }
///       """)]
///   public partial void AddVectors(CudaBuffer&lt;float&gt; a, CudaBuffer&lt;float&gt; b, CudaBuffer&lt;float&gt; c);
/// </code>
///
/// Usage – external .cu file:
/// <code>
///   [GpuKernel(SourceFile = "Kernels/flash_attn.cu")]
///   public partial void FlashAttn(CudaBuffer&lt;float&gt; q, CudaBuffer&lt;float&gt; k, CudaBuffer&lt;float&gt; v, CudaBuffer&lt;float&gt; o);
/// </code>
///
/// Usage – runtime compilation (no build-time CUDA toolchain needed):
/// <code>
///   [GpuKernel("...", Compilation = KernelCompilation.Runtime)]
///   public partial void MyKernel(CudaBuffer&lt;float&gt; b);
/// </code>
/// </summary>
/// <param name="source">
/// Inline CUDA C/C++ source. Use a raw string literal to avoid escaping:
/// <c>[GpuKernel("""...""")]</c>
/// </param>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class GpuKernelAttribute(string source = "") : Attribute
{
    /// <summary>
    /// Inline CUDA C/C++ source code for this kernel.
    /// The C# method name must exactly match the <c>extern "C" __global__</c> function name.
    /// Use a C# 11 raw string literal (<c>"""..."""</c>) to avoid escaping double quotes.
    /// </summary>
    public string Source { get; } = source;

    /// <summary>
    /// Path to an external <c>.cu</c> file, relative to the project root.
    /// Mutually exclusive with the <see cref="Source"/> constructor argument.
    /// </summary>
    public string SourceFile { get; init; } = string.Empty;

    /// <summary>
    /// Compile only this single architecture for this kernel, overriding the project-wide
    /// <c>KernelSharpMinArch</c> setting. Leave empty (default) to use the project default.
    /// Examples: <c>"compute_80"</c>, <c>"sm_89"</c>, <c>"80"</c>.
    /// For <see cref="KernelCompilation.BuildTime"/> this is the PTX virtual arch floor.
    /// For <see cref="KernelCompilation.Runtime"/> leave empty to auto-detect the native SM.
    /// </summary>
    public string Arch { get; init; } = string.Empty;

    /// <summary>
    /// Additional NVRTC options appended verbatim for this kernel only, space-separated.
    /// Examples: <c>"-DMYMACRO=1"</c>, <c>"-lineinfo"</c>.
    /// Project-wide extra options live in <c>KernelSharpExtraOptions</c> (Directory.Build.props).
    /// </summary>
    public string ExtraFlags { get; init; } = string.Empty;

    /// <summary>
    /// Additional include path passed to NVRTC for this kernel only, overriding the
    /// project-wide <c>KernelSharpIncludePath</c> setting.
    /// </summary>
    public string IncludePath { get; init; } = string.Empty;

    /// <summary>
    /// When <see langword="true"/>, skips compilation entirely and emits
    /// <c>throw new NotImplementedException()</c> as the method body.  Use this to stub
    /// out kernels that are not yet ready, so the rest of the project still compiles fast.
    /// </summary>
    public bool NotImplemented { get; init; } = false;

    /// <summary>
    /// Per-kernel PTX embedding compression override.
    /// Valid values: <c>"brotli"</c>, <c>"gzip"</c>, <c>"zlib"</c>, <c>"deflate"</c>, <c>"none"</c>.
    /// Leave empty (default) to use the project-wide <c>KernelSharpPtxCompression</c>
    /// MSBuild property.  Only applies to <see cref="KernelCompilation.BuildTime"/> kernels.
    /// </summary>
    public string Compression { get; init; } = "";

    /// <summary>
    /// Controls when the CUDA source is compiled to PTX.
    /// <list type="bullet">
    ///   <item><see cref="KernelCompilation.BuildTime"/> (default) — NVRTC at build time;
    ///   PTX is embedded in the assembly.</item>
    ///   <item><see cref="KernelCompilation.Runtime"/> — NVRTC at first kernel call;
    ///   CUDA source is embedded as a string, native SM is detected automatically.</item>
    /// </list>
    /// </summary>
    public KernelCompilation Compilation { get; init; } = KernelCompilation.BuildTime;

    /// <summary>
    /// Number of CUDA threads per block used in the generated <c>cuLaunchKernel</c> call.
    /// 0 (default) lets the generator use 256.
    /// Must be a multiple of 32 and ≤ 1024.
    /// </summary>
    public int ThreadsPerBlock { get; init; } = 0;

    /// <summary>
    /// Fixed number of blocks in the X-dimension of the grid.
    /// 0 (default) auto-computes as <c>ceil(n / ThreadsPerBlock)</c>.
    /// Set to 1 for single-block kernels such as inclusive prefix scans.
    /// </summary>
    public int BlocksPerGrid { get; init; } = 0;
}