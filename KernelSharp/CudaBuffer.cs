namespace KernelSharp;

/// <summary>
/// A typed, contiguous block of CUDA device memory holding elements of type <typeparamref name="T"/>.
/// The buffer does <b>not</b> own the memory by default; use <see cref="Allocate"/> to create
/// an owning buffer that is freed on <see cref="Dispose"/>.
/// </summary>
public sealed class CudaBuffer<T> : IDisposable where T : unmanaged
{
    private IntPtr _devicePointer;
    private readonly bool _ownsMemory;
    private bool _disposed;

    /// <summary>Pointer to the device allocation.</summary>
    public IntPtr DevicePointer
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _devicePointer;
        }
    }

    /// <summary>Number of <typeparamref name="T"/> elements the buffer can hold.</summary>
    public int Length { get; }

    /// <summary>Size of the allocation in bytes (<c>Length * sizeof(T)</c>).</summary>
    public unsafe long ByteSize => (long)Length * sizeof(T);

    private CudaBuffer(IntPtr ptr, int length, bool owns)
    {
        _devicePointer = ptr;
        Length = length;
        _ownsMemory = owns;
    }

    /// <summary>
    /// Wrap an existing device pointer as a non-owning view.
    /// The caller is responsible for the memory lifetime.
    /// </summary>
    public static CudaBuffer<T> FromPointer(IntPtr devicePtr, int elementCount)
        => new(devicePtr, elementCount, owns: false);

    /// <summary>
    /// Allocate <paramref name="elementCount"/> elements of <typeparamref name="T"/> on the current CUDA device.
    /// </summary>
    public static unsafe CudaBuffer<T> Allocate(int elementCount)
    {
        long bytes = (long)elementCount * sizeof(T);
        CudaDriverApi.CheckResult(CudaDriverApi.cuMemAlloc(out IntPtr ptr, new IntPtr(bytes)));
        return new(ptr, elementCount, owns: true);
    }

    /// <summary>Copy a managed array to device memory (blocking, host→device).</summary>
    public unsafe void CopyFromHost(T[] source)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(source);
        fixed (T* ptr = source)
        {
            CudaDriverApi.CheckResult(
                CudaDriverApi.cuMemcpyHtoD(_devicePointer, (IntPtr)ptr,
                    new IntPtr((long)source.Length * sizeof(T))));
        }
    }

    /// <summary>Copy a span to device memory (blocking, host→device).</summary>
    public unsafe void CopyFromHost(ReadOnlySpan<T> source)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        fixed (T* ptr = source)
        {
            CudaDriverApi.CheckResult(
                CudaDriverApi.cuMemcpyHtoD(_devicePointer, (IntPtr)ptr,
                    new IntPtr((long)source.Length * sizeof(T))));
        }
    }

    /// <summary>Copy device memory back to a managed array (blocking, device→host).</summary>
    public unsafe void CopyToHost(T[] destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(destination);
        fixed (T* ptr = destination)
        {
            CudaDriverApi.CheckResult(
                CudaDriverApi.cuMemcpyDtoH((IntPtr)ptr, _devicePointer,
                    new IntPtr((long)destination.Length * sizeof(T))));
        }
    }

    /// <summary>Copy device memory back to a span (blocking, device→host).</summary>
    public unsafe void CopyToHost(Span<T> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        fixed (T* ptr = destination)
        {
            CudaDriverApi.CheckResult(
                CudaDriverApi.cuMemcpyDtoH((IntPtr)ptr, _devicePointer,
                    new IntPtr((long)destination.Length * sizeof(T))));
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        if (_ownsMemory && _devicePointer != IntPtr.Zero)
        {
            CudaDriverApi.cuMemFree(_devicePointer);
            _devicePointer = IntPtr.Zero;
        }
    }
}