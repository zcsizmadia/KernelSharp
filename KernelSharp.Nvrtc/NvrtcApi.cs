using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

// When compiled into KernelSharp.Build.dll, this class lives in the Build namespace
// under the name NvrtcCompiler to avoid a type-identity conflict with the public
// KernelSharp.NvrtcApi exposed by KernelSharp.dll.  The MSBuild task references it
// as NvrtcCompiler; the generated per-kernel code calls the public KernelSharp.NvrtcApi.
#if KERNELSHARP_BUILD_TASK
namespace KernelSharp.Build;
#else
namespace KernelSharp;
#endif

/// <summary>Result codes returned by NVRTC API functions.</summary>
internal enum NvrtcResult
{
    Success = 0,
    OutOfMemory = 1,
    ProgramCreationFailure = 2,
    InvalidInput = 3,
    InvalidProgram = 4,
    InvalidOption = 5,
    CompilationError = 6,
    BuiltinOperationFailure = 7,
    NoNameExpressionsAfterCompilation = 8,
    NoLoweredNamesBeforeCompilation = 9,
    NameExpressionNotValid = 10,
    InternalError = 11,
}

/// <summary>
/// Dynamically-loaded bindings for the NVRTC library (nvrtc64_*.dll / libnvrtc.so).
/// Compiles CUDA C/C++ source to PTX in-process — no subprocess, no temp files.
/// Thread-safe: multiple threads may call <see cref="Compile"/> concurrently.
/// </summary>
#if KERNELSHARP_BUILD_TASK
internal static class NvrtcCompiler
#else
public static partial class NvrtcApi
#endif
{
    private static IntPtr _lib = IntPtr.Zero;
    private static readonly object _initLock = new();

    // ── Delegate types for each NVRTC entry point ─────────────────────────

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate NvrtcResult CreateProgramDelegate(
        out IntPtr prog,
        [MarshalAs(UnmanagedType.LPStr)] string src,
        [MarshalAs(UnmanagedType.LPStr)] string name,
        int numHeaders, IntPtr headers, IntPtr includeNames);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate NvrtcResult CompileProgramDelegate(
        IntPtr prog, int numOptions,
        [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] options);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate NvrtcResult GetPTXSizeDelegate(IntPtr prog, out nuint ptxSizeRet);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate NvrtcResult GetPTXDelegate(IntPtr prog, [Out] byte[] ptx);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate NvrtcResult DestroyProgramDelegate(ref IntPtr prog);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate NvrtcResult GetProgramLogSizeDelegate(IntPtr prog, out nuint logSizeRet);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate NvrtcResult GetProgramLogDelegate(IntPtr prog, [Out] byte[] log);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate NvrtcResult VersionDelegate(out int major, out int minor);

    // ── Loaded function pointers ──────────────────────────────────────────

    private static CreateProgramDelegate?      _createProgram;
    private static CompileProgramDelegate?     _compileProgram;
    private static GetPTXSizeDelegate?         _getPTXSize;
    private static GetPTXDelegate?             _getPTX;
    private static DestroyProgramDelegate?     _destroyProgram;
    private static GetProgramLogSizeDelegate?  _getProgramLogSize;
    private static GetProgramLogDelegate?      _getProgramLog;
    private static VersionDelegate?            _version;

    // ── Initialisation ────────────────────────────────────────────────────

    /// <summary>
    /// Loads the NVRTC library if not already loaded.
    /// Throws <see cref="InvalidOperationException"/> if the library cannot be found.
    /// </summary>
    /// <param name="customCudaRoot">
    /// Optional explicit CUDA installation root (overrides env vars).
    /// When <see langword="null"/>, the standard search order is used:
    /// <c>KERNELSHARP_CUDA_PATH</c>, <c>CUDA_PATH</c>, <c>CUDA_TOOLKIT_ROOT_DIR</c>, PATH.
    /// </param>
#if KERNELSHARP_BUILD_TASK
    internal
#else
    public
#endif
    static void EnsureLoaded(string? customCudaRoot = null)
    {
        if (_lib != IntPtr.Zero) return;
        lock (_initLock)
        {
            if (_lib != IntPtr.Zero) return;
            string path = FindNvrtcLibrary(customCudaRoot);
            if (string.IsNullOrEmpty(path))
                throw new InvalidOperationException(
                    "NVRTC library not found. Install the CUDA Toolkit and ensure CUDA_PATH is set, " +
                    "or set KERNELSHARP_CUDA_PATH to your CUDA installation root. " +
                    "On Linux, ensure libnvrtc.so is in /usr/local/cuda/lib64 or LD_LIBRARY_PATH.");

            // NVRTC internally loads companion DLLs (e.g. nvrtc-builtins64_NNN.dll)
            // using a bare filename, so they must be findable via the OS search path.
            // Prepend the NVRTC directory to PATH / LD_LIBRARY_PATH so that works
            // regardless of whether the user has the CUDA bin/ dir on their system PATH.
            string nvrtcDir = Path.GetDirectoryName(path)!;
            PrependLibrarySearchPath(nvrtcDir);

            _lib = NativeLibrary.Load(path);
            _createProgram     = GetExport<CreateProgramDelegate>    ("nvrtcCreateProgram");
            _compileProgram    = GetExport<CompileProgramDelegate>   ("nvrtcCompileProgram");
            _getPTXSize        = GetExport<GetPTXSizeDelegate>       ("nvrtcGetPTXSize");
            _getPTX            = GetExport<GetPTXDelegate>           ("nvrtcGetPTX");
            _destroyProgram    = GetExport<DestroyProgramDelegate>   ("nvrtcDestroyProgram");
            _getProgramLogSize = GetExport<GetProgramLogSizeDelegate>("nvrtcGetProgramLogSize");
            _getProgramLog     = GetExport<GetProgramLogDelegate>    ("nvrtcGetProgramLog");
            _version           = GetExport<VersionDelegate>          ("nvrtcVersion");
        }
    }

    private static T GetExport<T>(string name) where T : Delegate =>
        Marshal.GetDelegateForFunctionPointer<T>(NativeLibrary.GetExport(_lib, name));

    /// <summary>
    /// Prepends <paramref name="dir"/> to the OS DLL search path so that
    /// companion libraries loaded by NVRTC (e.g. nvrtc-builtins64_NNN.dll)
    /// are found without requiring the user to have CUDA on their system PATH.
    /// </summary>
    private static void PrependLibrarySearchPath(string dir)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            // AddDllDirectory registers an additional search directory for
            // LoadLibrary calls made by any DLL in the process.
            AddDllDirectory(dir);

            // Also prepend to PATH as a fallback for older Windows / .NET versions.
            string path = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
            if (!path.Contains(dir, StringComparison.OrdinalIgnoreCase))
                Environment.SetEnvironmentVariable("PATH", dir + Path.PathSeparator + path);
        }
        else
        {
            // On Linux/macOS NVRTC searches LD_LIBRARY_PATH for companion .so files.
            const string ldVar = "LD_LIBRARY_PATH";
            string existing = Environment.GetEnvironmentVariable(ldVar) ?? string.Empty;
            if (!existing.Contains(dir, StringComparison.Ordinal))
                Environment.SetEnvironmentVariable(ldVar, dir + Path.PathSeparator + existing);
        }
    }

    [DllImport("kernel32", ExactSpelling = true, SetLastError = true)]
    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    private static extern IntPtr AddDllDirectory([MarshalAs(UnmanagedType.LPWStr)] string newDirectory);

    // ── Version ───────────────────────────────────────────────────────────

    /// <summary>Returns the NVRTC version string, e.g. "12.4", or empty if not loaded.</summary>
#if KERNELSHARP_BUILD_TASK
    internal
#else
    public
#endif
    static string GetVersion()
    {
        if (_lib == IntPtr.Zero) return string.Empty;
        try { _version!(out int major, out int minor); return $"{major}.{minor}"; }
        catch { return string.Empty; }
    }

    // ── Compilation ───────────────────────────────────────────────────────

    /// <summary>
    /// Compiles CUDA C/C++ source to PTX using NVRTC.
    /// </summary>
    /// <param name="cudaSource">CUDA C/C++ source code (UTF-8).</param>
    /// <param name="arch">
    /// Target architecture, e.g. <c>"compute_80"</c>, <c>"sm_89"</c>, or bare <c>"80"</c>.
    /// Defaults to <c>"compute_75"</c> when empty.
    /// </param>
    /// <param name="extraOptions">
    /// Additional NVRTC options, space-separated (e.g. <c>"-DMYMACRO=1 -lineinfo"</c>).
    /// </param>
    /// <param name="includePath">Additional include directory passed as <c>-I</c>.</param>
    /// <returns>PTX bytes (null-terminated UTF-8 string) ready for <c>cuModuleLoadData</c>.</returns>
#if KERNELSHARP_BUILD_TASK
    internal
#else
    public
#endif
    static byte[] Compile(
        string cudaSource,
        string arch,
        string extraOptions = "",
        string includePath  = "")
    {
        EnsureLoaded();

        string effectiveArch = string.IsNullOrWhiteSpace(arch) ? "compute_75" : arch;
        string archOption    = NormalizeArchOption(effectiveArch);

        var options = new List<string>
        {
            archOption,
            "--std=c++17",
            "--use_fast_math",
            "-D_USE_MATH_DEFINES",
            "-DNDEBUG",
        };

        if (!string.IsNullOrWhiteSpace(includePath))
            options.Add($"-I{includePath.TrimEnd('\\', '/')}");

        if (!string.IsNullOrWhiteSpace(extraOptions))
            options.AddRange(extraOptions.Split(
                new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries));

        var result = _createProgram!(
            out IntPtr prog, cudaSource, "kernel.cu", 0, IntPtr.Zero, IntPtr.Zero);
        CheckResult(result, "nvrtcCreateProgram");

        try
        {
            result = _compileProgram!(prog, options.Count, options.ToArray());
            if (result != NvrtcResult.Success)
            {
                string log = GetLog(prog);
                throw new InvalidOperationException($"NVRTC compilation failed:\n{log}");
            }

            result = _getPTXSize!(prog, out nuint ptxSize);
            CheckResult(result, "nvrtcGetPTXSize");

            byte[] ptx = new byte[(int)ptxSize];
            result = _getPTX!(prog, ptx);
            CheckResult(result, "nvrtcGetPTX");

            return ptx;
        }
        finally
        {
            _destroyProgram!(ref prog);
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static string GetLog(IntPtr prog)
    {
        try
        {
            _getProgramLogSize!(prog, out nuint sz);
            byte[] buf = new byte[(int)sz];
            _getProgramLog!(prog, buf);
            return Encoding.UTF8.GetString(buf).TrimEnd('\0');
        }
        catch { return "(could not retrieve compilation log)"; }
    }

    private static void CheckResult(NvrtcResult result, string op)
    {
        if (result != NvrtcResult.Success)
            throw new InvalidOperationException($"NVRTC error in {op}: {result}");
    }

    /// <summary>
    /// Converts bare arch strings to the <c>--gpu-architecture=…</c> NVRTC option.
    /// Examples: <c>"80"</c> → <c>"--gpu-architecture=compute_80"</c>;
    ///           <c>"compute_89"</c> → <c>"--gpu-architecture=compute_89"</c>;
    ///           <c>"sm_90"</c> → <c>"--gpu-architecture=sm_90"</c>
    /// </summary>
    internal static string NormalizeArchOption(string arch)
    {
        arch = arch.Trim();
        if (arch.StartsWith("--gpu-architecture=", StringComparison.Ordinal)) return arch;
        if (arch.StartsWith("compute_", StringComparison.Ordinal)
            || arch.StartsWith("sm_", StringComparison.Ordinal))
            return $"--gpu-architecture={arch}";
        return $"--gpu-architecture=compute_{arch.Replace(".", string.Empty)}";
    }

    // ── Library discovery ─────────────────────────────────────────────────

    /// <summary>
    /// Searches for the NVRTC shared library.
    /// Priority: <paramref name="customCudaRoot"/>, <c>KERNELSHARP_CUDA_PATH</c>,
    /// <c>CUDA_PATH</c>, <c>CUDA_TOOLKIT_ROOT_DIR</c>, PATH, standard install paths.
    /// </summary>
#if KERNELSHARP_BUILD_TASK
    internal
#else
    public
#endif
    static string FindNvrtcLibrary(string? customCudaRoot = null)
    {
        bool isWin = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

        var roots = new List<string?>();
        if (!string.IsNullOrWhiteSpace(customCudaRoot)) roots.Add(customCudaRoot);
        roots.Add(Environment.GetEnvironmentVariable("KERNELSHARP_CUDA_PATH"));
        roots.Add(Environment.GetEnvironmentVariable("CUDA_PATH"));
        roots.Add(Environment.GetEnvironmentVariable("CUDA_TOOLKIT_ROOT_DIR"));

        foreach (string? root in roots)
        {
            if (string.IsNullOrEmpty(root)) continue;
            string? hit = FindInCudaRoot(root, isWin);
            if (hit != null) return hit;
        }

        string? pathEnv = Environment.GetEnvironmentVariable("PATH");
        if (pathEnv != null)
        {
            foreach (string entry in pathEnv.Split(Path.PathSeparator))
            {
                string dir = entry.Trim();
                if (isWin)
                {
                    string? hit = FindNvrtcDllInDir(dir);
                    if (hit != null) return hit;
                }
                else
                {
                    // entry might be a bin/ dir → check ../lib64 and ../lib
                    string parent = Path.GetFullPath(Path.Combine(dir, ".."));
                    string? hit = FindLinuxLib(Path.Combine(parent, "lib64"))
                               ?? FindLinuxLib(Path.Combine(parent, "lib"))
                               ?? FindLinuxLib(dir); // entry itself might be the lib dir
                    if (hit != null) return hit;
                }
            }
        }

        if (isWin)
        {
            string pf = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
            string nvidiaCuda = Path.Combine(pf, "NVIDIA GPU Computing Toolkit", "CUDA");
            if (Directory.Exists(nvidiaCuda))
            {
                string[] vers = Directory.GetDirectories(nvidiaCuda);
                Array.Sort(vers, StringComparer.OrdinalIgnoreCase);
                Array.Reverse(vers);
                foreach (string ver in vers)
                {
                    string? hit = FindNvrtcDllInDir(Path.Combine(ver, "bin"));
                    if (hit != null) return hit;
                }
            }
        }
        else
        {
            foreach (string lib in new[]
            {
                "/usr/local/cuda/lib64",
                "/usr/local/cuda/lib",
                "/usr/lib/x86_64-linux-gnu",
            })
            {
                string? hit = FindLinuxLib(lib);
                if (hit != null) return hit;
            }
        }

        return string.Empty;
    }

    private static string? FindInCudaRoot(string root, bool isWin)
    {
        if (isWin)
            return FindNvrtcDllInDir(Path.Combine(root, "bin"));

        // conda installs to lib/ not lib64/; check both
        return FindLinuxLib(Path.Combine(root, "lib64"))
            ?? FindLinuxLib(Path.Combine(root, "lib"));
    }

    private static string? FindNvrtcDllInDir(string dir)
    {
        if (!Directory.Exists(dir)) return null;
        string[] candidates = Directory.GetFiles(dir, "nvrtc64_*.dll");
        if (candidates.Length == 0) return null;
        Array.Sort(candidates, StringComparer.OrdinalIgnoreCase);
        Array.Reverse(candidates);
        return candidates[0];
    }

    private static string? FindLinuxLib(string lib64)
    {
        if (!Directory.Exists(lib64)) return null;
        string symlink = Path.Combine(lib64, "libnvrtc.so");
        if (File.Exists(symlink)) return symlink;
        string[] versioned = Directory.GetFiles(lib64, "libnvrtc.so.*");
        if (versioned.Length == 0) return null;
        Array.Sort(versioned, StringComparer.OrdinalIgnoreCase);
        Array.Reverse(versioned);
        return versioned[0];
    }
}
