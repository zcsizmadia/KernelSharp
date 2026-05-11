using KernelSharp;

namespace KernelSharp.Samples;

/// <summary>
/// Sample – kernel source loaded from an external .cu file via
/// <c>[GpuKernel(SourceFile = "…")]</c>.
/// <para>
/// Instead of embedding CUDA C++ as an inline raw string literal, you can point
/// to a <c>.cu</c> file on disk.  The path is resolved relative to the C# source
/// file that declares the method, so <c>"ClampKernel.cu"</c> here refers to
/// <c>Kernels/ClampKernel.cu</c> sitting next to this file.
/// </para>
/// </summary>
public partial class ExternalFileSample
{
    /// <summary>
    /// Clamps every element of <paramref name="x"/> into
    /// [<paramref name="lo"/>, <paramref name="hi"/>] and writes the result to
    /// <paramref name="y"/>.  The kernel source lives in
    /// <c>ClampKernel.cu</c> rather than an inline string.
    /// </summary>
    [GpuKernel(SourceFile = "ClampKernel.cu")]
    public partial void ClampVector(
        CudaBuffer<float> x, CudaBuffer<float> y, float lo, float hi, int n);
}
