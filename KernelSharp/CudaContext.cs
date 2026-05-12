namespace KernelSharp;

/// <summary>
/// Manages the CUDA Driver API context lifecycle.
/// Call <see cref="Initialize"/> once at application startup before using any GPU kernels.
/// </summary>
public sealed class CudaContext : IDisposable
{
    private IntPtr _ctx;
    private bool _disposed;

    private CudaContext(IntPtr ctx) => _ctx = ctx;

    /// <summary>The raw <c>CUcontext</c> handle.</summary>
    public IntPtr Handle
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _ctx;
        }
    }

    /// <summary>
    /// Initialise the CUDA Driver API and create a primary context on
    /// <paramref name="deviceOrdinal"/> (default: device 0).
    /// </summary>
    public static CudaContext Initialize(int deviceOrdinal = 0)
    {
        CudaDriverApi.CheckResult(CudaDriverApi.cuInit(0));
        CudaDriverApi.CheckResult(CudaDriverApi.cuDeviceGet(out int dev, deviceOrdinal));
        CudaDriverApi.CheckResult(CudaDriverApi.cuCtxCreate(out IntPtr ctx, 0, dev));
        return new CudaContext(ctx);
    }

    /// <summary>
    /// Returns the number of CUDA-capable devices on this machine.
    /// Returns 0 when no CUDA driver is installed.
    /// </summary>
    public static int DeviceCount()
    {
        try
        {
            CudaDriverApi.CheckResult(CudaDriverApi.cuInit(0));
            CudaDriverApi.CheckResult(CudaDriverApi.cuDeviceGetCount(out int count));
            return count;
        }
        catch (DllNotFoundException) { return 0; }
        catch (CudaException) { return 0; }
    }

    /// <summary>
    /// Makes this context current on the calling thread.
    /// CUDA Driver API contexts are thread-local: call this at the start of every
    /// thread (or <c>[Before(Test)]</c> hook) that will issue CUDA operations.
    /// </summary>
    public void MakeCurrent()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CudaDriverApi.CheckResult(CudaDriverApi.cuCtxSetCurrent(_ctx));
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        if (_ctx != IntPtr.Zero)
        {
            CudaDriverApi.cuCtxDestroy(_ctx);
            _ctx = IntPtr.Zero;
        }
    }
}