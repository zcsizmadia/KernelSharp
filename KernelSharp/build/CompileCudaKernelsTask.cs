using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;

using Microsoft.Build.Framework;
using Microsoft.Build.Utilities;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

using ParallelOptions = System.Threading.Tasks.ParallelOptions;

namespace KernelSharp.Build;

/// <summary>
/// MSBuild task that discovers [GpuKernel] partial methods in C# source files, compiles
/// the inline or file-referenced CUDA source with nvcc in parallel, and writes per-kernel
/// C# launcher files that embed the fatbinary and the CUDA Driver API dispatch code.
///
/// Runs before CoreCompile. Generated files are added to @(Compile) via the target's
/// Output element so the Roslyn compiler sees them automatically.
/// </summary>
public sealed class CompileCudaKernelsTask : Microsoft.Build.Utilities.Task, ICancelableTask
{
    // ── Cancellation (ICancelable / Ctrl+C) ───────────────────────────────────

    private readonly CancellationTokenSource _cts = new();

    /// <summary>
    /// Registered live nvcc processes so they can be killed immediately on Cancel().
    /// Processes are removed once they exit so the set never grows unboundedly.
    /// </summary>
    private readonly ConcurrentDictionary<int, Process> _activeProcesses = new();

    /// <summary>
    /// Called by MSBuild when the user presses Ctrl+C or the build is cancelled.
    /// Kills every nvcc process that is still running so none survive as orphans.
    /// </summary>
    public void Cancel()
    {
        _cts.Cancel();
        foreach (var p in _activeProcesses.Values)
        {
            try
            {
                if (!p.HasExited)
                {
                    p.Kill(entireProcessTree: true);
                }
            }
            catch
            {
                // best-effort
            }
        }
    }

    // ── Inputs ───────────────────────────────────────────────────────────────

    /// <summary>All C# source files in the project (@(Compile)).</summary>
    [Required]
    public ITaskItem[] CompileItems { get; set; } = [];

    /// <summary>Path to the CUDA include directory (cuda.h, cuda_runtime.h, …).</summary>
    public string IncludePath { get; set; } = string.Empty;

    /// <summary>C++ standard passed to nvcc (default: c++20).</summary>
    public string NvccStd { get; set; } = "c++20";

    /// <summary>Extra nvcc flags appended to every kernel in this project.</summary>
    public string NvccExtraFlags { get; set; } = string.Empty;

    /// <summary>Windows only: path to cl.exe used as nvcc host compiler.</summary>
    public string MsvcClPath { get; set; } = string.Empty;

    /// <summary>Comma-separated GPU arch list, e.g. "compute_80,compute_89,compute_90".</summary>
    public string TargetArchs { get; set; } = "compute_80,compute_89,compute_90";

    /// <summary>Max parallel nvcc processes. Empty = all CPU cores.</summary>
    public string MaxParallelism { get; set; } = string.Empty;

    /// <summary>
    /// Verbosity level for KernelSharp messages: "normal" (default) or "detailed".
    /// "detailed" promotes nvcc command lines and up-to-date skips to Normal importance
    /// so they appear without having to raise the overall MSBuild verbosity.
    /// </summary>
    public string Verbosity { get; set; } = "normal";

    /// <summary>Fatbin embedding compression: "gzip" (default) or "none".</summary>
    public string FatbinCompression { get; set; } = "gzip";

    /// <summary>$(IntermediateOutputPath) — fallback folder for generated .cs files.</summary>
    [Required]
    public string IntermediateOutputPath { get; set; } = string.Empty;

    /// <summary>
    /// Optional committed-source folder for generated .cs files.
    /// When set, files are written here instead of $(IntermediateOutputPath)/KernelSharp/,
    /// allowing them to be checked into source control so that build machines
    /// without nvcc can compile using pre-built launchers.
    /// nvcc is still skipped whenever the generated file is newer than the source.
    /// </summary>
    public string GeneratedOutputPath { get; set; } = string.Empty;

    // ── Outputs ──────────────────────────────────────────────────────────────

    /// <summary>Generated .cs launcher files to add to @(Compile).</summary>
    [Output]
    public ITaskItem[] GeneratedFiles { get; set; } = [];

    // ── Entry point ──────────────────────────────────────────────────────────

    public override bool Execute()
    {
        string outDir = !string.IsNullOrWhiteSpace(GeneratedOutputPath)
            ? GeneratedOutputPath.TrimEnd('\\', '/')
            : Path.Combine(IntermediateOutputPath.TrimEnd('\\', '/'), "KernelSharp");
        Directory.CreateDirectory(outDir);

        // 1. Parse all .cs files syntactically to find [GpuKernel] methods
        var kernels = CollectKernels();

        // Emit build warnings for any signature mismatches detected at parse time
        foreach (var k in kernels)
        {
            if (k.ValidationWarning != null)
            {
                Log.LogWarning(null, "KERNELSHARP003", null, k.SourceFilePath, 0, 0, 0, 0,
                    $"KernelSharp: {k.ValidationWarning}");
            }
        }

        if (kernels.Count == 0)
        {
            Log.LogMessage(MessageImportance.Low, "KernelSharp: no [GpuKernel] kernels found in project.");
            return true;
        }

        // 2. Separate up-to-date kernels from those needing (re)compilation.
        //    A kernel is up-to-date when its generated .cs is newer than its source .cs
        //    AND the cached file contains an actual fatbin (not a stale NotImplemented stub).
        var toCompile = new List<KernelInfo>();
        var upToDatePaths = new List<string>();

        bool isDetailed = string.Equals(Verbosity, "detailed", StringComparison.OrdinalIgnoreCase);
        MessageImportance verboseImportance = isDetailed ? MessageImportance.Normal : MessageImportance.Low;

        foreach (var k in kernels)
        {
            string genPath = Path.Combine(outDir, $"{k.ClassName}.{k.MethodName}.g.cs");
            if (!k.NotImplemented && IsUpToDate(k.SourceFilePath, genPath) && !IsStubFile(genPath))
            {
                upToDatePaths.Add(genPath);
                Log.LogMessage(verboseImportance,
                    $"KernelSharp:   {k.ClassName}.{k.MethodName} — up-to-date, skipping nvcc.");
            }
            else
            {
                toCompile.Add(k);
            }
        }

        // 3. Locate nvcc and cl.exe (done once, before the parallel loop)
        string nvcc = FindNvcc();
        string clDir = IsWindows ? FindClDir(MsvcClPath) : string.Empty;

        if (string.IsNullOrEmpty(nvcc) && toCompile.Any(k => !k.NotImplemented))
        {
            Log.LogWarning(null, "KERNELSHARP001", null, null, 0, 0, 0, 0,
                "KernelSharp: nvcc not found — CUDA kernel compilation skipped. " +
                "Set CUDA_PATH or ensure nvcc is on PATH. Kernels will fail at runtime.");
        }

        // 4. Log a summary line at Normal importance (hidden at Minimal)
        if (toCompile.Count > 0)
        {
            CompilerInfo ci = string.IsNullOrEmpty(nvcc)
                ? CompilerInfo.Empty
                : CompilerInfo.Query(nvcc, clDir);
            Log.LogMessage(MessageImportance.Normal,
                $"KernelSharp: compiling {toCompile.Count} CUDA kernel(s)" +
                (ci == CompilerInfo.Empty ? string.Empty : $" [{ci}]"));
        }

        // 5. Compile kernels in parallel (thread-safe — each uses its own temp dir)
        CompilerInfo compInfo = string.IsNullOrEmpty(nvcc)
            ? CompilerInfo.Empty
            : CompilerInfo.Query(nvcc, clDir);

        int maxPar = ParseMaxParallelism(MaxParallelism);
        var results = new KernelResult[toCompile.Count];
        System.Threading.Tasks.Parallel.For(0, toCompile.Count,
            new ParallelOptions { MaxDegreeOfParallelism = maxPar },
            i =>
            {
                var k = toCompile[i];
                if (k.NotImplemented || string.IsNullOrEmpty(nvcc))
                {
                    results[i] = new KernelResult(k, null, string.Empty, null);
                    return;
                }
                results[i] = CompileKernel(k, nvcc, clDir, verboseImportance);
            });

        // 6. Write generated .cs files and collect paths for @(Compile)
        bool success = true;
        var generatedPaths = new List<string>(upToDatePaths);

        for (int i = 0; i < results.Length; i++)
        {
            var r = results[i];

            if (r.Error != null)
            {
                Log.LogError(null, "KERNELSHARP002", null, r.Kernel.SourceFilePath, 0, 0, 0, 0,
                    $"KernelSharp: nvcc failed for '{r.Kernel.ClassName}.{r.Kernel.MethodName}': {r.Error}");
                success = false;
            }
            else
            {
                string detail = r.Kernel.NotImplemented ? " — stub (NotImplemented)"
                    : r.FatbinBytes != null ? $" — {r.FatbinBytes.Length} bytes fatbin"
                    : " — nvcc skipped (not found)";
                Log.LogMessage(MessageImportance.Normal,
                    $"KernelSharp:   {r.Kernel.ClassName}.{r.Kernel.MethodName}{detail}");
            }

            string genPath = Path.Combine(outDir, $"{r.Kernel.ClassName}.{r.Kernel.MethodName}.g.cs");
            string cs = BuildLauncherSource(r.Kernel, r.FatbinBytes,
                EffectiveCompression(r.Kernel.Compression, FatbinCompression),
                compInfo, r.NvccArgs);
            File.WriteAllText(genPath, cs, Encoding.UTF8);
            generatedPaths.Add(genPath);
        }

        GeneratedFiles = [.. generatedPaths.Select(p => (ITaskItem)new TaskItem(p))];

        if (GeneratedFiles.Length > 0)
        {
            Log.LogMessage(MessageImportance.Normal,
                $"KernelSharp: {GeneratedFiles.Length} kernel launcher file(s) ready" +
                (upToDatePaths.Count > 0 ? $" ({upToDatePaths.Count} from cache)" : string.Empty) + ".");
        }

        return success;
    }

    // ── Kernel discovery (syntactic Roslyn parsing) ───────────────────────────

    private List<KernelInfo> CollectKernels()
    {
        var result = new List<KernelInfo>();
        foreach (var item in CompileItems)
        {
            string path = item.GetMetadata("FullPath");
            if (string.IsNullOrEmpty(path) || !path.EndsWith(".cs", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (!File.Exists(path))
            {
                continue;
            }

            string source;
            try { source = File.ReadAllText(path, Encoding.UTF8); }
            catch { continue; }

            // Fast pre-screen: skip files that don't reference [GpuKernel] at all
            if (!source.Contains("GpuKernel", StringComparison.Ordinal))
            {
                continue;
            }

            SyntaxTree tree = CSharpSyntaxTree.ParseText(source, new CSharpParseOptions(LanguageVersion.Latest));
            ParseKernels(tree, path, result);
        }
        return result;
    }

    internal static void ParseKernels(SyntaxTree tree, string filePath, List<KernelInfo> kernels)
    {
        var root = tree.GetRoot();
        foreach (var method in root.DescendantNodes().OfType<MethodDeclarationSyntax>())
        {
            if (!method.Modifiers.Any(m => m.IsKind(SyntaxKind.PartialKeyword)))
            {
                continue;
            }

            if (method.Parent is not ClassDeclarationSyntax cls)
            {
                continue;
            }

            if (!cls.Modifiers.Any(m => m.IsKind(SyntaxKind.PartialKeyword)))
            {
                continue;
            }

            // Find [GpuKernel] or [GpuKernelAttribute] on this method
            AttributeSyntax? gpuAttr = null;
            foreach (var attrList in method.AttributeLists)
            {
                foreach (var attr in attrList.Attributes)
                {
                    string name = attr.Name.ToString();
                    if (name is "GpuKernel" or "GpuKernelAttribute"
                        || name.EndsWith(".GpuKernel", StringComparison.Ordinal)
                        || name.EndsWith(".GpuKernelAttribute", StringComparison.Ordinal))
                    {
                        gpuAttr = attr;
                        break;
                    }
                }
                if (gpuAttr != null)
                {
                    break;
                }
            }
            if (gpuAttr == null)
            {
                continue;
            }

            // Extract attribute arguments
            string kernelSource = string.Empty;
            string sourceFile = string.Empty;
            string arch = string.Empty;
            string extraFlags = string.Empty;
            string incPath = string.Empty;
            string compression = string.Empty;
            bool notImpl = false;

            var args = gpuAttr.ArgumentList?.Arguments ?? default;
            if (args.Count > 0 && args[0].NameEquals == null)
            {
                kernelSource = ExtractStringLiteral(args[0].Expression);
            }

            foreach (var arg in args)
            {
                if (arg.NameEquals == null)
                {
                    continue;
                }

                switch (arg.NameEquals.Name.Identifier.Text)
                {
                    case "SourceFile": sourceFile = ExtractStringLiteral(arg.Expression); break;
                    case "Arch": arch = ExtractStringLiteral(arg.Expression); break;
                    case "ExtraFlags": extraFlags = ExtractStringLiteral(arg.Expression); break;
                    case "IncludePath": incPath = ExtractStringLiteral(arg.Expression); break;
                    case "Compression": compression = ExtractStringLiteral(arg.Expression); break;
                    case "NotImplemented": notImpl = ExtractBoolLiteral(arg.Expression); break;
                }
            }

            // Resolve SourceFile relative to the .cs file
            if (string.IsNullOrWhiteSpace(kernelSource) && !string.IsNullOrWhiteSpace(sourceFile))
            {
                string? dir = Path.GetDirectoryName(filePath);
                string fullPath = !string.IsNullOrEmpty(dir)
                    ? Path.GetFullPath(Path.Combine(dir, sourceFile))
                    : sourceFile;
                if (File.Exists(fullPath))
                {
                    kernelSource = File.ReadAllText(fullPath, Encoding.UTF8);
                }
            }

            if (string.IsNullOrWhiteSpace(kernelSource) && !notImpl)
            {
                continue;
            }

            string ns = GetNamespace(cls);
            string className = cls.Identifier.Text;
            string methodName = method.Identifier.Text;
            string paramList = method.ParameterList.ToString();

            var @params = method.ParameterList.Parameters
                .Select(p => new KernelParam(
                    p.Identifier.Text,
                    p.Type?.ToString() ?? string.Empty,
                    (p.Type?.ToString() ?? string.Empty).StartsWith("CudaBuffer", StringComparison.Ordinal)))
                .ToArray();

            // Validate the CUDA source signature against the C# declaration
            string cudaFuncName = methodName;  // default: assume names match by convention
            string? validationWarn = null;
            if (!string.IsNullOrEmpty(kernelSource))
            {
                var sig = ExtractCudaSignature(kernelSource);
                if (sig == null)
                {
                    validationWarn = $"No '__global__' function found in kernel source for '{className}.{methodName}'. " +
                                     "Make sure the CUDA source contains 'extern \"C\" __global__ void {methodName}(…)'";
                }
                else
                {
                    cudaFuncName = sig.Value.Name;
                    var warnings = new List<string>();
                    if (!string.Equals(cudaFuncName, methodName, StringComparison.Ordinal))
                    {
                        warnings.Add($"CUDA function name '{cudaFuncName}' does not match C# method name '{methodName}' " +
                                     $"in '{className}'. '{cudaFuncName}' will be used for cuModuleGetFunction.");
                    }

                    int csCount = @params.Length;
                    int cudaCount = sig.Value.ParamCount;
                    // Allow +1 for auto-injected 'n' (added when no int param exists and buffers are present)
                    bool autoN = csCount == cudaCount - 1 && @params.Any(p => p.IsBuffer) && !@params.Any(p => !p.IsBuffer && p.TypeSyntax == "int");
                    if (cudaCount != csCount && !autoN)
                    {
                        warnings.Add($"CUDA function '{cudaFuncName}' has {cudaCount} parameter(s) but C# method '{methodName}' " +
                                     $"has {csCount}. Verify the signatures match.");
                    }

                    if (warnings.Count > 0)
                    {
                        validationWarn = string.Join(" ", warnings);
                    }
                }
            }

            kernels.Add(new KernelInfo(
                filePath, ns, className, methodName, cudaFuncName,
                kernelSource, arch, extraFlags, incPath,
                paramList, @params, notImpl, compression, validationWarn));
        }
    }

    private static string ExtractStringLiteral(ExpressionSyntax expr) =>
        expr is LiteralExpressionSyntax lit && lit.Token.Value is string s ? s : string.Empty;

    private static bool ExtractBoolLiteral(ExpressionSyntax expr) =>
        expr is LiteralExpressionSyntax lit && lit.Token.Value is bool b && b;

    private static string GetNamespace(ClassDeclarationSyntax cls)
    {
        var parts = new List<string>();
        SyntaxNode? node = cls.Parent;
        while (node != null)
        {
            if (node is NamespaceDeclarationSyntax ns)
            {
                parts.Insert(0, ns.Name.ToString());
            }
            else if (node is FileScopedNamespaceDeclarationSyntax fns)
            {
                parts.Insert(0, fns.Name.ToString());
            }

            node = node.Parent;
        }
        return string.Join(".", parts);
    }

    // ── Incremental build check ───────────────────────────────────────────────

    internal static bool IsUpToDate(string sourceFile, string generatedFile)
    {
        if (!File.Exists(generatedFile))
        {
            return false;
        }

        try { return File.GetLastWriteTimeUtc(generatedFile) >= File.GetLastWriteTimeUtc(sourceFile); }
        catch { return false; }
    }

    /// <summary>
    /// Returns true when a generated .g.cs is a NotImplemented stub — i.e. it contains no
    /// compiled fatbin.  A stale stub must not be treated as up-to-date for a kernel that is
    /// no longer flagged NotImplemented in the C# source.
    /// </summary>
    internal static bool IsStubFile(string generatedFile)
    {
        try
        {
            // Read only the first 2 KB — the stub marker appears near the top.
            using var fs = new FileStream(generatedFile, FileMode.Open, FileAccess.Read, FileShare.Read);
            int toRead = (int)Math.Min(fs.Length, 2048);
            byte[] buf = new byte[toRead];
            int read = fs.Read(buf, 0, toRead);
            string header = Encoding.UTF8.GetString(buf, 0, read);
            return header.Contains("NotImplementedException", StringComparison.Ordinal);
        }
        catch { return false; }
    }

    // ── nvcc invocation (thread-safe, each call uses its own temp dir) ────────

    private KernelResult CompileKernel(KernelInfo k, string nvcc, string clDir, MessageImportance verboseImportance)
    {
        string effectiveInc = !string.IsNullOrWhiteSpace(k.IncludePath) ? k.IncludePath
            : !string.IsNullOrWhiteSpace(IncludePath) ? IncludePath
            : Environment.GetEnvironmentVariable("CUDA_INCLUDE_PATH") ?? string.Empty;

        string archOverride = string.IsNullOrWhiteSpace(k.Arch) ? string.Empty : NormalizeArch(k.Arch);
        string[] archs = string.IsNullOrEmpty(archOverride)
            ? [.. (TargetArchs ?? "compute_80")
                .Split(new[] { ';', ',', ' ' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(NormalizeArch).Distinct()]
            : [archOverride];
        if (archs.Length == 0)
        {
            archs = ["compute_80"];
        }

        string tempDir = Path.Combine(Path.GetTempPath(), "KernelSharpGen",
            $"{k.MethodName}_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        string srcFile = Path.Combine(tempDir, $"{k.MethodName}.cu");
        string fatbinFile = Path.Combine(tempDir, $"{k.MethodName}.fatbin");
        File.WriteAllText(srcFile, k.KernelSource, Encoding.UTF8);

        bool isWin = IsWindows;
        char sep = isWin ? '\\' : '/';
        var sb = new StringBuilder();

        string ccbinPrefix = isWin && !string.IsNullOrEmpty(clDir)
            ? $"-ccbin \"{clDir}\" " : string.Empty;
        if (!string.IsNullOrEmpty(ccbinPrefix))
        {
            sb.Append(ccbinPrefix);
        }

        sb.Append("-fatbin ");
        string? lowestArch = null;
        foreach (string arch in archs)
        {
            string num = arch.Replace("compute_", string.Empty);
            sb.Append($"-gencode arch={arch},code=sm_{num} ");
            lowestArch ??= arch;
        }
        if (lowestArch != null)
        {
            sb.Append($"-gencode arch={lowestArch},code={lowestArch} ");
        }

        string std = string.IsNullOrWhiteSpace(NvccStd) ? "c++20" : NvccStd;
        sb.Append($"-x cu -std={std} --extended-lambda --use_fast_math ");

        string cudaRoot = GetCudaRoot(nvcc);
        if (string.IsNullOrEmpty(cudaRoot))
        {
            cudaRoot = Environment.GetEnvironmentVariable("CUDA_PATH") ?? string.Empty;
        }

        if (!string.IsNullOrWhiteSpace(effectiveInc))
        {
            string inc = effectiveInc.TrimEnd('\\', '/');
            sb.Append($"-I\"{inc}\" ");
        }
        if (!string.IsNullOrEmpty(cudaRoot))
        {
            sb.Append($"-I\"{cudaRoot}{sep}include\" ");
            string cccl13 = Path.Combine(cudaRoot, "include", "cccl");
            if (Directory.Exists(cccl13))
            {
                sb.Append($"-I\"{cccl13}\" ");
            }
        }

        sb.Append("-D_USE_MATH_DEFINES ");
        if (isWin)
        {
            sb.Append("-D_WINDOWS -DFMT_UNICODE=0 -DNDEBUG -DNOMINMAX -DWIN32_LEAN_AND_MEAN -DNOGDI ");
            sb.Append("-Xcompiler \"/EHsc /Zc:__cplusplus /utf-8 /wd4996 /wd4100 /wd4864 /wd4702 /wd4324 /wd4714 /Zc:preprocessor /WX\" ");
        }
        else
        {
            sb.Append("-Xcompiler \"-fPIC\" ");
        }

        if (!string.IsNullOrWhiteSpace(NvccExtraFlags))
        {
            sb.Append($"{NvccExtraFlags} ");
        }

        if (!string.IsNullOrWhiteSpace(k.ExtraFlags))
        {
            sb.Append($"{k.ExtraFlags} ");
        }


        sb.Append($"\"{srcFile}\" -o \"{fatbinFile}\"");

        var nvccArgs = sb.ToString();

        // Log the nvcc command
        Log.LogCommandLine(verboseImportance, $"nvcc {nvccArgs}");

        var proc = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = nvcc,
                Arguments = nvccArgs,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            }
        };

        if (!proc.Start())
        {
            TryDelete(tempDir);
            return new KernelResult(k, null, nvccArgs, "Could not start nvcc process.");
        }

        _activeProcesses.TryAdd(proc.Id, proc);

        string stderr = proc.StandardError.ReadToEnd();
        string stdout = proc.StandardOutput.ReadToEnd();

        // Poll WaitForExit so we react to cancellation promptly.
        while (!proc.WaitForExit(500))
        {
            if (_cts.IsCancellationRequested)
            {
                try
                {
                    if (!proc.HasExited)
                    {
                        proc.Kill(entireProcessTree: true);
                    }
                }
                catch
                {
                    // best-effort
                }

                _activeProcesses.TryRemove(proc.Id, out _);
                TryDelete(tempDir);
                return new KernelResult(k, null, nvccArgs, "Compilation cancelled.");
            }
        }

        int exitCode = proc.ExitCode;
        _activeProcesses.TryRemove(proc.Id, out _);
        proc.Dispose();

        if (exitCode != 0)
        {
            TryDelete(tempDir);
            return new KernelResult(k, null, nvccArgs, (stderr + stdout).Trim());
        }

        byte[]? fatbin = File.Exists(fatbinFile) ? File.ReadAllBytes(fatbinFile) : null;
        TryDelete(tempDir);
        return new KernelResult(k, fatbin, nvccArgs, null);
    }

    private static void TryDelete(string dir) { try { Directory.Delete(dir, recursive: true); } catch { } }

    // ── C# launcher source code emission ─────────────────────────────────────

    internal static string BuildLauncherSource(
        KernelInfo k, byte[]? fatbin, string compression,
        CompilerInfo ci, string nvccArgs)
    {
        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine($"// Kernel   : {k.ClassName}.{k.MethodName}");
        if (!string.IsNullOrEmpty(ci.CudaVersion))
        {
            sb.AppendLine($"// CUDA     : {ci.CudaVersion}");
        }

        if (!string.IsNullOrEmpty(ci.NvccVersion))
        {
            sb.AppendLine($"// nvcc     : {ci.NvccVersion}");
        }

        if (!string.IsNullOrEmpty(ci.HostCompiler))
        {
            sb.AppendLine($"// Compiler : {ci.HostCompiler}");
        }

        if (!string.IsNullOrEmpty(nvccArgs))
        {
            sb.AppendLine($"// nvcc args: {nvccArgs}");
        }

        sb.AppendLine("#nullable enable");
        sb.AppendLine("using System;");
        sb.AppendLine("using KernelSharp;");
        sb.AppendLine();

        if (!string.IsNullOrEmpty(k.Namespace))
        {
            sb.AppendLine($"namespace {k.Namespace};");
            sb.AppendLine();
        }

        sb.AppendLine($"partial class {k.ClassName}");
        sb.AppendLine("{");

        if (k.NotImplemented)
        {
            sb.AppendLine($"    public partial void {k.MethodName}{k.ParameterList}");
            sb.AppendLine("        => throw new NotImplementedException(");
            sb.AppendLine($"            $\"{k.ClassName}.{k.MethodName} is marked [GpuKernel(NotImplemented=true)]\");");
            sb.AppendLine();
            sb.AppendLine("}");
            return sb.ToString();
        }

        bool useGzip = compression == "gzip";
        int rawLen = fatbin?.Length ?? 0;
        byte[]? embed = rawLen > 0 && useGzip ? GZipCompress(fatbin!) : fatbin;
        int embedLen = embed?.Length ?? 0;

        sb.AppendLine("    // Compression format used when embedding the fatbin at build time.");
        sb.AppendLine($"    private const string _{k.MethodName}_compression = \"{compression}\";");
        sb.AppendLine();

        if (embedLen > 0)
        {
            sb.AppendLine(useGzip
                ? $"    // fatbin for '{k.MethodName}' — {rawLen} bytes raw, {embedLen} bytes gzip-compressed, embedded at build time"
                : $"    // fatbin for '{k.MethodName}' — {rawLen} bytes, embedded uncompressed at build time");
            sb.AppendLine($"    private static readonly byte[] _{k.MethodName}_fatbin_encoded =");
            sb.AppendLine("        Convert.FromBase64String(");
            EmitBase64Chunks(sb, Convert.ToBase64String(embed!), "            ");
        }
        else
        {
            sb.AppendLine($"    private static readonly byte[] _{k.MethodName}_fatbin_encoded = Array.Empty<byte>();");
        }
        sb.AppendLine();

        sb.AppendLine("#pragma warning disable CS0162 // Unreachable code — intentional: compression constant is baked in at build time");
        sb.AppendLine($"    private static readonly byte[] _{k.MethodName}_fatbin =");
        sb.AppendLine($"        global::KernelSharp.FatbinHelper.Decode(_{k.MethodName}_fatbin_encoded, _{k.MethodName}_compression);");
        sb.AppendLine("#pragma warning restore CS0162");
        sb.AppendLine();
        sb.AppendLine($"    private static IntPtr _{k.MethodName}_module = IntPtr.Zero;");
        sb.AppendLine($"    private static IntPtr _{k.MethodName}_func   = IntPtr.Zero;");
        sb.AppendLine();

        sb.AppendLine($"    private static void {k.MethodName}_EnsureLoaded()");
        sb.AppendLine("    {");
        sb.AppendLine($"        if (_{k.MethodName}_module != IntPtr.Zero) return;");
        sb.AppendLine("        unsafe");
        sb.AppendLine("        {");
        sb.AppendLine($"            fixed (byte* _p = _{k.MethodName}_fatbin)");
        sb.AppendLine("            {");
        sb.AppendLine("                CudaDriverApi.CheckResult(");
        sb.AppendLine($"                    CudaDriverApi.cuModuleLoadData(out _{k.MethodName}_module, (IntPtr)_p));");
        sb.AppendLine("                CudaDriverApi.CheckResult(");
        sb.AppendLine($"                    CudaDriverApi.cuModuleGetFunction(out _{k.MethodName}_func, _{k.MethodName}_module, \"{k.CudaFunctionName}\"));");
        sb.AppendLine("            }");
        sb.AppendLine("        }");
        sb.AppendLine("    }");
        sb.AppendLine();

        var bufferParams = k.Params.Where(p => p.IsBuffer).ToArray();
        var firstIntScalar = k.Params.FirstOrDefault(p => !p.IsBuffer && p.TypeSyntax == "int");
        bool autoInjectN = firstIntScalar == null && bufferParams.Length > 0;
        int kpSize = k.Params.Length + (autoInjectN ? 1 : 0);

        sb.AppendLine($"    public partial void {k.MethodName}{k.ParameterList}");
        sb.AppendLine("    {");
        sb.AppendLine($"        {k.MethodName}_EnsureLoaded();");
        sb.AppendLine("        unsafe");
        sb.AppendLine("        {");
        for (int i = 0; i < k.Params.Length; i++)
        {
            var p = k.Params[i];
            sb.AppendLine(p.IsBuffer
                ? $"            IntPtr _p{i} = {p.Name}.DevicePointer;"
                : $"            {p.TypeSyntax} _p{i} = {p.Name};");
        }
        sb.AppendLine($"            void** _kp = stackalloc void*[{kpSize}];");
        for (int i = 0; i < k.Params.Length; i++)
        {
            sb.AppendLine($"            _kp[{i}] = (void*)(&_p{i});");
        }

        string nExpr = firstIntScalar != null ? firstIntScalar.Name
            : bufferParams.Length > 0 ? $"(int){bufferParams[0].Name}.Length" : "1";
        sb.AppendLine($"            int _n = {nExpr};");
        if (autoInjectN)
        {
            sb.AppendLine($"            _kp[{k.Params.Length}] = (void*)(&_n);  // auto-injected n");
        }

        sb.AppendLine("            uint _threads = 256;");
        sb.AppendLine("            uint _blocks = (uint)((_n + (int)_threads - 1) / (int)_threads);");
        sb.AppendLine();
        sb.AppendLine($"            CudaDriverApi.CheckResult(CudaDriverApi.cuLaunchKernel(");
        sb.AppendLine($"                _{k.MethodName}_func,");
        sb.AppendLine("                _blocks, 1, 1, _threads, 1, 1,");
        sb.AppendLine("                0, IntPtr.Zero, _kp, null));");
        sb.AppendLine("        }");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("}");
        return sb.ToString();
    }

    // Matches the first __global__ function signature in a CUDA source string.
    // Group 1 = function name, Group 2 = raw parameter list (may be empty).
    private static readonly Regex s_globalKernelRx = new(
        @"__global__\s+[\w\s*]+?\s+(\w+)\s*\(([^)]*?)\)",
        RegexOptions.Compiled | RegexOptions.Singleline);

    /// <summary>
    /// Extracts the first <c>__global__</c> function name and parameter count from a CUDA source string.
    /// Returns <c>null</c> if no <c>__global__</c> function is found.
    /// </summary>
    internal static (string Name, int ParamCount)? ExtractCudaSignature(string source)
    {
        var m = s_globalKernelRx.Match(source);
        if (!m.Success)
        {
            return null;
        }

        string name = m.Groups[1].Value;
        string paramBlock = m.Groups[2].Value.Trim();
        int paramCount = string.IsNullOrEmpty(paramBlock) ? 0 : paramBlock.Split(',').Length;
        return (name, paramCount);
    }

    internal static void EmitBase64Chunks(StringBuilder sb, string b64, string indent)
    {
        const int chunkSize = 128;
        for (int start = 0; start < b64.Length; start += chunkSize)
        {
            int len = Math.Min(chunkSize, b64.Length - start);
            string chunk = b64.Substring(start, len);
            bool last = start + len >= b64.Length;
            sb.AppendLine(last ? $"{indent}\"{chunk}\");" : $"{indent}\"{chunk}\" +");
        }
    }

    private static byte[] GZipCompress(byte[] data)
    {
        var ms = new MemoryStream();
        using (var gz = new GZipStream(ms, CompressionLevel.Optimal, leaveOpen: true))
        {
            gz.Write(data, 0, data.Length);
        }

        return ms.ToArray();
    }

    internal static string EffectiveCompression(string attrOverride, string projectDefault) =>
        string.IsNullOrWhiteSpace(attrOverride)
            ? projectDefault
            : attrOverride.Trim().ToLowerInvariant();

    // ── Toolchain discovery ───────────────────────────────────────────────────

    private static string FindNvcc()
    {
        bool isWin = IsWindows;
        string exe = isWin ? "nvcc.exe" : "nvcc";

        foreach (string ev in new[] { "CUDA_PATH", "CUDA_TOOLKIT_ROOT_DIR" })
        {
            string? root = Environment.GetEnvironmentVariable(ev);
            if (!string.IsNullOrEmpty(root))
            {
                string c = Path.Combine(root, "bin", exe);
                if (File.Exists(c))
                {
                    return c;
                }
            }
        }

        string? pathEnv = Environment.GetEnvironmentVariable("PATH");
        if (pathEnv != null)
        {
            foreach (string d in pathEnv.Split(Path.PathSeparator))
            {
                string c = Path.Combine(d.Trim(), exe);
                if (File.Exists(c))
                {
                    return c;
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
                    string c = Path.Combine(ver, "bin", exe);
                    if (File.Exists(c))
                    {
                        return c;
                    }
                }
            }
        }
        else
        {
            foreach (string prefix in new[] { "/usr/local/cuda/bin", "/usr/bin" })
            {
                string c = Path.Combine(prefix, exe);
                if (File.Exists(c))
                {
                    return c;
                }
            }
        }

        return string.Empty;
    }

    private static string FindClDir(string explicitClPath)
    {
        if (!string.IsNullOrWhiteSpace(explicitClPath))
        {
            string c = explicitClPath.Trim('"').TrimEnd('\\', '/');
            if (File.Exists(c))
            {
                return Path.GetDirectoryName(c) ?? string.Empty;
            }

            if (Directory.Exists(c))
            {
                return c;
            }
        }

        string? vcTools = Environment.GetEnvironmentVariable("VCToolsInstallDir");
        if (!string.IsNullOrEmpty(vcTools))
        {
            string dir = Path.Combine(vcTools.TrimEnd('\\', '/'), "bin", "HostX64", "x64");
            if (Directory.Exists(dir))
            {
                return dir;
            }
        }

        string pf86 = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86);
        string vswhere = Path.Combine(pf86, "Microsoft Visual Studio", "Installer", "vswhere.exe");
        if (File.Exists(vswhere)) { string? d = TryFindClViaVswhere(vswhere); if (d != null)
            {
                return d;
            }
        }

        string? d2 = ScanVsRootsForCl();
        if (d2 != null)
        {
            return d2;
        }

        string? pathEnv = Environment.GetEnvironmentVariable("PATH");
        if (pathEnv != null)
        {
            foreach (string entry in pathEnv.Split(Path.PathSeparator))
            {
                if (File.Exists(Path.Combine(entry.Trim(), "cl.exe")))
                {
                    return entry.Trim();
                }
            }
        }

        return string.Empty;
    }

    private static string? TryFindClViaVswhere(string vswhere)
    {
        try
        {
            var p = Process.Start(new ProcessStartInfo
            {
                FileName = vswhere,
                Arguments = "-latest -prerelease -products * -property installationPath",
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            })!;
            string output = p.StandardOutput.ReadToEnd().Trim();
            p.WaitForExit();
            p.Dispose();
            foreach (string line in output.Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries))
            {
                string? d = FindClInVsRoot(line.Trim());
                if (d != null)
                {
                    return d;
                }
            }
        }
        catch { }
        return null;
    }

    private static string? ScanVsRootsForCl()
    {
        foreach (Environment.SpecialFolder sf in new[]
            { Environment.SpecialFolder.ProgramFiles, Environment.SpecialFolder.ProgramFilesX86 })
        {
            string vsRoot = Path.Combine(Environment.GetFolderPath(sf), "Microsoft Visual Studio");
            if (!Directory.Exists(vsRoot))
            {
                continue;
            }

            string[] majors = Directory.GetDirectories(vsRoot);
            Array.Sort(majors, StringComparer.OrdinalIgnoreCase);
            Array.Reverse(majors);
            foreach (string major in majors)
            {
                string[] eds = Directory.GetDirectories(major);
                Array.Sort(eds, StringComparer.OrdinalIgnoreCase);
                Array.Reverse(eds);
                foreach (string ed in eds) { string? d = FindClInVsRoot(ed); if (d != null)
                    {
                        return d;
                    }
                }
            }
        }
        return null;
    }

    private static string? FindClInVsRoot(string vsRoot)
    {
        string msvcRoot = Path.Combine(vsRoot, "VC", "Tools", "MSVC");
        if (!Directory.Exists(msvcRoot))
        {
            return null;
        }

        string[] versions = Directory.GetDirectories(msvcRoot);
        Array.Sort(versions, StringComparer.OrdinalIgnoreCase);
        Array.Reverse(versions);
        foreach (string ver in versions)
        {
            string dir = Path.Combine(ver, "bin", "HostX64", "x64");
            if (File.Exists(Path.Combine(dir, "cl.exe")))
            {
                return dir;
            }
        }
        return null;
    }

    internal static string GetCudaRoot(string nvcc)
    {
        string nvccDir = Path.GetDirectoryName(nvcc) ?? string.Empty;
        return Path.GetDirectoryName(nvccDir) ?? string.Empty;
    }

    internal static string NormalizeArch(string arch)
    {
        arch = arch.Trim();
        if (arch.StartsWith("compute_", StringComparison.Ordinal))
        {
            return arch;
        }

        if (arch.StartsWith("sm_", StringComparison.Ordinal))
        {
            return "compute_" + arch[3..];
        }

        return arch.Contains('.') ? "compute_" + arch.Replace(".", string.Empty) : "compute_" + arch;
    }

    internal static int ParseMaxParallelism(string value) =>
        !string.IsNullOrWhiteSpace(value)
        && int.TryParse(value.Trim(), out int parsed) && parsed > 0
            ? parsed : -1;  // -1 = all cores (Parallel.For default)

    private static bool IsWindows =>
        System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
            System.Runtime.InteropServices.OSPlatform.Windows);

    // ── Compiler version query ────────────────────────────────────────────────

    internal sealed class CompilerInfo
    {
        public static readonly CompilerInfo Empty = new(string.Empty, string.Empty, string.Empty);

        public string CudaVersion { get; }
        public string NvccVersion { get; }
        public string HostCompiler { get; }

        private CompilerInfo(string cuda, string nvcc, string host)
        {
            CudaVersion = cuda;
            NvccVersion = nvcc;
            HostCompiler = host;
        }

        public override string ToString() =>
            string.Join(", ", new[] { NvccVersion, HostCompiler }
                .Where(s => !string.IsNullOrEmpty(s)));

        internal static CompilerInfo Query(string nvcc, string clDir)
        {
            string cudaVer = string.Empty, nvccVer = string.Empty, hostVer = string.Empty;

            if (!string.IsNullOrEmpty(nvcc))
            {
                string nvccOut = RunProc(nvcc, "--version", stderr: false);
                foreach (string line in nvccOut.Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries))
                {
                    int ri = line.IndexOf("release ", StringComparison.OrdinalIgnoreCase);
                    int vi = line.IndexOf(", V", StringComparison.OrdinalIgnoreCase);
                    if (ri >= 0 && vi > ri)
                    {
                        cudaVer = line.Substring(ri + 8, vi - ri - 8).Trim();
                        nvccVer = line[(vi + 3)..].Trim();
                        break;
                    }
                }
            }

            if (IsWindows && !string.IsNullOrEmpty(clDir))
            {
                string clExe = Path.Combine(clDir, "cl.exe");
                if (File.Exists(clExe))
                {
                    string firstLine = RunProc(clExe, string.Empty, stderr: true)
                        .Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries)
                        .FirstOrDefault() ?? string.Empty;
                    int verIdx = firstLine.IndexOf("Version ", StringComparison.OrdinalIgnoreCase);
                    if (verIdx >= 0)
                    {
                        hostVer = "MSVC " + firstLine[(verIdx + 8)..].Trim().Replace(" for ", " ");
                    }
                    else if (firstLine.Length > 0)
                    {
                        hostVer = "MSVC " + firstLine;
                    }
                }
            }
            else if (!IsWindows)
            {
                string firstLine = RunProc("gcc", "--version", stderr: false)
                    .Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries)
                    .FirstOrDefault() ?? string.Empty;
                if (firstLine.Length > 0)
                {
                    string[] parts = firstLine.Split(' ');
                    hostVer = "gcc " + (parts.Length > 0 ? parts[^1].Trim() : firstLine);
                }
            }

            return new CompilerInfo(cudaVer, nvccVer, hostVer);
        }

        private static string RunProc(string exe, string args, bool stderr)
        {
            try
            {
                var p = Process.Start(new ProcessStartInfo
                {
                    FileName = exe,
                    Arguments = args,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                })!;
                string stdout = p.StandardOutput.ReadToEnd();
                string stderrTx = p.StandardError.ReadToEnd();
                p.WaitForExit();
                p.Dispose();
                return stderr ? stderrTx : stdout;
            }
            catch { return string.Empty; }
        }
    }

    // ── Data models ───────────────────────────────────────────────────────────

    internal sealed class KernelInfo(
        string sourceFilePath, string ns, string className, string methodName, string cudaFunctionName,
        string kernelSource, string arch, string extraFlags, string incPath,
        string paramList, CompileCudaKernelsTask.KernelParam[] @params, bool notImpl, string compression,
        string? validationWarning = null)
    {
        public string SourceFilePath { get; } = sourceFilePath; public string Namespace { get; } = ns; public string ClassName { get; } = className;
        /// <summary>C# partial method name.</summary>
        public string MethodName { get; } = methodName;         /// <summary>
                                                                /// Actual <c>__global__</c> function name extracted from the CUDA source.
                                                                /// Defaults to <see cref="MethodName"/> when no parseable signature is found.
                                                                /// Used verbatim in <c>cuModuleGetFunction</c>.
                                                                /// </summary>
        public string CudaFunctionName { get; } = cudaFunctionName;
        public string KernelSource { get; } = kernelSource; public string Arch { get; } = arch;
        public string ExtraFlags { get; } = extraFlags; public string IncludePath { get; } = incPath; public string ParameterList { get; } = paramList;
        public KernelParam[] Params { get; } = @params; public bool NotImplemented { get; } = notImpl; public string Compression { get; } = compression;
        /// <summary>Non-null when a name or arg-count mismatch was detected at parse time.</summary>
        public string? ValidationWarning { get; } = validationWarning;
    }

    internal sealed class KernelParam
    {
        public string Name { get; }
        public string TypeSyntax { get; }
        public bool IsBuffer { get; }
        public KernelParam(string name, string typeSyntax, bool isBuffer)
            => (Name, TypeSyntax, IsBuffer) = (name, typeSyntax, isBuffer);
    }

    private sealed class KernelResult
    {
        public KernelInfo Kernel { get; }
        public byte[]? FatbinBytes { get; }
        public string NvccArgs { get; }
        public string? Error { get; }
        public KernelResult(KernelInfo kernel, byte[]? fatbin, string nvccArgs, string? error)
            => (Kernel, FatbinBytes, NvccArgs, Error) = (kernel, fatbin, nvccArgs, error);
    }
}