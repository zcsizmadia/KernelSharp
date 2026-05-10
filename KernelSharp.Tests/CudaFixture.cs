using KernelSharp;

using TUnit.Core;
using TUnit.Core.Exceptions;

namespace KernelSharp.Tests;

/// <summary>
/// Helper that checks whether a usable CUDA device is present.
/// Tests that require CUDA are skipped when none is available (CI without GPU).
/// </summary>
internal static class CudaFixture
{
    private static readonly Lazy<bool> _hasCuda = new(() =>
    {
        try { return CudaContext.DeviceCount() > 0; }
        catch { return false; }
    });

    public static bool HasCuda => _hasCuda.Value;

    /// <summary>
    /// Returns a new <see cref="CudaContext"/> or throws <see cref="SkipTestException"/>
    /// if no GPU is available.
    /// </summary>
    public static CudaContext RequireCuda()
    {
        if (!HasCuda)
            throw new SkipTestException("No CUDA-capable GPU detected – test skipped.");
        return CudaContext.Initialize();
    }
}