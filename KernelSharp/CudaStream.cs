namespace KernelSharp;

/// <summary>
/// Wraps a <c>CUstream</c> handle for asynchronous kernel execution.
/// </summary>
public sealed class CudaStream : IDisposable
{
    private IntPtr _stream;
    private bool _disposed;

    /// <summary>The raw <c>CUstream</c> handle (zero = default stream).</summary>
    public IntPtr Handle
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _stream;
        }
    }

    private CudaStream(IntPtr handle) => _stream = handle;

    /// <summary>Create a new asynchronous CUDA stream.</summary>
    public static CudaStream Create()
    {
        CudaDriverApi.CheckResult(CudaDriverApi.cuStreamCreate(out IntPtr s, 0));
        return new CudaStream(s);
    }

    /// <summary>Represents the default (null) stream.</summary>
    public static CudaStream Default { get; } = new(IntPtr.Zero);

    /// <summary>Block the calling thread until all work on this stream completes.</summary>
    public void Synchronize()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        CudaDriverApi.CheckResult(CudaDriverApi.cuStreamSynchronize(_stream));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_stream != IntPtr.Zero)
        {
            CudaDriverApi.cuStreamDestroy(_stream);
            _stream = IntPtr.Zero;
        }
    }
}