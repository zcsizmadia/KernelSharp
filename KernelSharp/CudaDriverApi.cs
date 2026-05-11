using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace KernelSharp;

/// <summary>
/// Minimal P/Invoke bindings for the CUDA Driver API (<c>nvcuda</c> / <c>libcuda</c>).
/// Only the entry points required by the kernel module loader are declared.
/// </summary>
public static partial class CudaDriverApi
{
    // Resolve the correct library name at class init.
    private const string LibName = "nvcuda"; // Windows: nvcuda.dll  Linux: libcuda.so.1

    // ── Initialisation ───────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial CuResult cuInit(uint flags);

    // ── Device ───────────────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial CuResult cuDeviceGet(out int device, int ordinal);

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial CuResult cuDeviceGetCount(out int count);


    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial CuResult cuDeviceGetAttribute(out int pi, CuDeviceAttribute attrib, int dev);
    // ── Context ──────────────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial CuResult cuCtxCreate(out IntPtr ctx, uint flags, int device);

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial CuResult cuCtxDestroy(IntPtr ctx);

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial CuResult cuCtxGetCurrent(out IntPtr ctx);

    // ── Module (fatbin / PTX load) ──────────────────────────────────────────

    /// <summary>
    /// Load a module from an in-memory image — accepts a fatbinary or a
    /// null-terminated PTX string.  The driver selects the best SASS code from
    /// the fatbin, or JIT-compiles PTX when only virtual arch code is present.
    /// </summary>
    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial CuResult cuModuleLoadData(out IntPtr module, IntPtr image);

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial CuResult cuModuleUnload(IntPtr module);

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial CuResult cuModuleGetFunction(out IntPtr hFunc, IntPtr hmod,
        [MarshalAs(UnmanagedType.LPStr)] string name);

    // ── Kernel launch ────────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static unsafe partial CuResult cuLaunchKernel(
        IntPtr f,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes,
        IntPtr hStream,
        void** kernelParams,
        void** extra);

    // ── Memory ───────────────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial CuResult cuMemAlloc(out IntPtr dptr, IntPtr bytesize);

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial CuResult cuMemFree(IntPtr dptr);

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial CuResult cuMemcpyHtoD(IntPtr dstDevice, IntPtr srcHost, IntPtr byteCount);

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial CuResult cuMemcpyDtoH(IntPtr dstHost, IntPtr srcDevice, IntPtr byteCount);

    // ── Stream ───────────────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial CuResult cuStreamCreate(out IntPtr stream, uint flags);

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial CuResult cuStreamSynchronize(IntPtr stream);

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial CuResult cuStreamDestroy(IntPtr stream);

    // ── Error ────────────────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial CuResult cuGetErrorString(CuResult error,
        out IntPtr pStr);

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Throws <see cref="CudaException"/> if <paramref name="result"/> is not
    /// <see cref="CuResult.Success"/>.
    /// </summary>
    public static void CheckResult(CuResult result)
    {
        if (result == CuResult.Success)
        {
            return;
        }

        string message = $"CUDA Driver error: {result}";
        if (cuGetErrorString(result, out IntPtr pStr) == CuResult.Success && pStr != IntPtr.Zero)
        {
            message = Marshal.PtrToStringAnsi(pStr) ?? message;
        }

        throw new CudaException(result, message);
    }
}

/// <summary>CUDA Driver API result codes (subset).</summary>
public enum CuResult : int
{
    Success = 0,
    ErrorInvalidValue = 1,
    ErrorOutOfMemory = 2,
    ErrorNotInitialized = 3,
    ErrorDeinitialized = 4,
    ErrorNoDevice = 100,
    ErrorInvalidDevice = 101,
    ErrorInvalidImage = 200,
    ErrorInvalidContext = 201,
    ErrorFileNotFound = 301,
    ErrorInvalidHandle = 400,
    ErrorNotFound = 500,
    ErrorNotReady = 600,
    ErrorLaunchFailed = 700,
    ErrorLaunchOutOfResources = 701,
    ErrorLaunchTimeout = 702,
    ErrorSharedObjectSymbolNotFound = 302,
    ErrorOperatingSystem = 304,
}

/// <summary>Exception thrown when a CUDA Driver API call fails.</summary>
public sealed class CudaException(CuResult result, string message)
    : Exception(message)
{
    /// <summary>The CUDA Driver result code that caused this exception.</summary>
    public CuResult Result { get; } = result;
}