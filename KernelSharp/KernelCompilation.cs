namespace KernelSharp;

/// <summary>
/// Controls when and how a <c>[GpuKernel]</c> method's CUDA source is compiled to PTX.
/// </summary>
public enum KernelCompilation
{
    /// <summary>
    /// Default. The NVRTC library compiles the CUDA source to PTX during <c>dotnet build</c>
    /// via the KernelSharp MSBuild task. The PTX is embedded in the assembly as a byte array.
    /// First kernel call loads the pre-compiled PTX with no JIT latency (beyond the driver's
    /// own one-time PTX→SASS translation, which is cached on disk).
    /// Requires NVRTC (<c>nvrtc64_*.dll</c> / <c>libnvrtc.so</c>) on the <em>build</em> machine.
    /// </summary>
    BuildTime,

    /// <summary>
    /// The CUDA C/C++ source is embedded as a string in the assembly. On the first kernel call,
    /// NVRTC detects the current GPU's compute capability and compiles the source to PTX optimised
    /// for that exact SM version. The driver caches the result so subsequent runs pay no JIT cost.
    /// Requires NVRTC on the <em>runtime</em> machine. No CUDA toolchain is needed at build time.
    /// Combine with <see cref="KernelPrewarmer"/> to push the compilation off the hot path.
    /// </summary>
    Runtime,
}
