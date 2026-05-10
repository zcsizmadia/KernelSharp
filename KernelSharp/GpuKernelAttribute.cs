namespace KernelSharp;

/// <summary>
/// Marks a <c>partial</c> method whose CUDA C/C++ implementation is provided inline
/// or via a source file.  The KernelSharp source generator invokes <c>nvcc</c> at
/// build time, embeds the resulting fatbinary directly in the assembly, and generates the
/// method body that loads and dispatches the kernel via the CUDA Driver API at runtime.
///
/// Usage – inline source (C# 11 raw string literal recommended):
/// <code>
///   [GpuKernel("""
///       extern "C" __global__ void AddVectors(
///           const float* a, const float* b, float* c, int n)
///       { int i = blockIdx.x*blockDim.x+threadIdx.x; if(i&lt;n) c[i]=a[i]+b[i]; }
///       """)]
///   public partial void AddVectors(CudaBuffer a, CudaBuffer b, CudaBuffer c);
/// </code>
///
/// Usage – external .cu file:
/// <code>
///   [GpuKernel(SourceFile = "Kernels/flash_attn.cu")]
///   public partial void FlashAttn(CudaBuffer q, CudaBuffer k, CudaBuffer v, CudaBuffer o);
/// </code>
/// </summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class GpuKernelAttribute : Attribute
{
    /// <summary>
    /// Inline CUDA C/C++ source code for this kernel.
    /// The C# method name must exactly match the <c>extern "C" __global__</c> function name.
    /// Use a C# 11 raw string literal (<c>"""..."""</c>) to avoid escaping double quotes.
    /// </summary>
    public string Source { get; }

    /// <summary>
    /// Path to an external <c>.cu</c> file, relative to the project root.
    /// Mutually exclusive with the <see cref="Source"/> constructor argument.
    /// </summary>
    public string SourceFile { get; init; } = string.Empty;

    /// <summary>
    /// Compile only this single architecture for this kernel, overriding the project-wide
    /// KernelSharpTargetArchs list. Leave empty (default) to compile all target archs.
    /// Examples: "compute_80", "sm_89", "80"
    /// </summary>
    public string Arch { get; init; } = string.Empty;

    /// <summary>
    /// Additional nvcc flags appended verbatim for this kernel only.
    /// Project-wide extra flags live in KernelSharpNvccExtraFlags (Directory.Build.props).
    /// </summary>
    public string ExtraFlags { get; init; } = string.Empty;

    /// <summary>
    /// Additional include path passed to nvcc for this kernel only, overriding the
    /// project-wide KernelSharpIncludePath setting.
    /// </summary>
    public string IncludePath { get; init; } = string.Empty;

    /// <summary>
    /// When <see langword="true"/>, the source generator skips nvcc entirely and emits
    /// <c>throw new NotImplementedException()</c> as the method body.  Use this to stub
    /// out kernels that are not yet ready, so the rest of the project still compiles fast.
    /// </summary>
    public bool NotImplemented { get; init; } = false;

    /// <summary>
    /// Per-kernel fatbin compression override.  Valid values: <c>"none"</c>, <c>"gzip"</c>.
    /// Leave empty (default) to use the project-wide <c>KernelSharpFatbinCompression</c>
    /// MSBuild property.  The compression format is stored as a constant in the generated
    /// source file so the loader always decodes correctly regardless of project settings.
    /// </summary>
    public string Compression { get; init; } = "";

    /// <param name="source">
    /// Inline CUDA C/C++ source. Use a raw string literal to avoid escaping:
    /// <c>[GpuKernel("""...""")]</c>
    /// </param>
    public GpuKernelAttribute(string source = "")
    {
        Source = source;
    }
}