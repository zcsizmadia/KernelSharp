namespace KernelSharp;

public static partial class NvrtcApi
{
    /// <summary>
    /// Detects the compute capability of the current CUDA device and returns it
    /// as an SM architecture string suitable for NVRTC, e.g. <c>"sm_89"</c>.
    /// Falls back to <c>"compute_75"</c> when no device is available or on error.
    /// Used by the runtime-compilation code path generated for
    /// <see cref="KernelCompilation.Runtime"/> kernels.
    /// </summary>
    public static string GetNativeArch()
    {
        try
        {
            CudaDriverApi.CheckResult(CudaDriverApi.cuDeviceGet(out int dev, 0));
            CudaDriverApi.CheckResult(CudaDriverApi.cuDeviceGetAttribute(
                out int major, CuDeviceAttribute.ComputeCapabilityMajor, dev));
            CudaDriverApi.CheckResult(CudaDriverApi.cuDeviceGetAttribute(
                out int minor, CuDeviceAttribute.ComputeCapabilityMinor, dev));
            return $"sm_{major}{minor}";
        }
        catch { return "compute_75"; }
    }
}
