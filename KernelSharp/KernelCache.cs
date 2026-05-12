using System.Collections.Concurrent;
using System.Text;

namespace KernelSharp;

/// <summary>
/// Thread-safe cache that holds loaded CUDA modules and their function handles.
/// The source generator emits calls to <see cref="GetOrLoadFunction"/> so that each
/// kernel module is loaded only once per process.
/// </summary>
public static class KernelCache
{
    private static readonly ConcurrentDictionary<string, (IntPtr Module, IntPtr Func)> _cache = new();

    /// <summary>
    /// Returns the <c>CUfunction</c> handle for <paramref name="functionName"/> embedded in
    /// <paramref name="moduleImage"/>, loading the module on first call.
    /// </summary>
    public static IntPtr GetOrLoadFunction(string moduleImage, string functionName)
    {
        string key = $"{functionName}@{moduleImage.GetHashCode()}";
        var entry = _cache.GetOrAdd(key, _ => LoadModule(moduleImage, functionName));
        return entry.Func;
    }

    private static unsafe (IntPtr Module, IntPtr Func) LoadModule(string image, string funcName)
    {
        byte[] bytes = Encoding.ASCII.GetBytes(image + "\0");
        IntPtr module;
        fixed (byte* p = bytes)
        {
            CudaDriverApi.CheckResult(
                CudaDriverApi.cuModuleLoadData(out module, (IntPtr)p));
        }

        CudaDriverApi.CheckResult(
            CudaDriverApi.cuModuleGetFunction(out IntPtr func, module, funcName));

        return (module, func);
    }

    /// <summary>
    /// Unloads all cached modules. Call at application shutdown if CUDA cleanup is needed.
    /// </summary>
    public static void Clear()
    {
        foreach (var (module, _) in _cache.Values)
        {
            CudaDriverApi.cuModuleUnload(module);
        }

        _cache.Clear();
    }
}