using System.Threading;

namespace KernelSharp;

/// <summary>
/// Provides background kernel pre-warming so the first real kernel launch does not pay
/// the module-load cost.  Call <see cref="Prewarm"/> once at application startup.
/// </summary>
public static class KernelPrewarmer
{
    /// <summary>
    /// Submits each (ModuleImage, FunctionName) pair to a background thread for loading.
    /// Returns immediately; loading happens concurrently.
    /// </summary>
    public static void Prewarm(IEnumerable<(string ModuleImage, string FunctionName)> kernels)
    {
        foreach (var (image, func) in kernels)
        {
            // Capture locals for closure
            string p = image;
            string f = func;
            ThreadPool.QueueUserWorkItem(_ =>
            {
                try { KernelCache.GetOrLoadFunction(p, f); }
                catch { /* Swallow – surfaced on first real use */ }
            });
        }
    }
}