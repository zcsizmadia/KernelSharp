using System;
using System.Collections.Generic;
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
/// the CUDA source in-process with NVRTC (no subprocess, no Visual Studio required), and
/// writes per-kernel C# launcher files that embed either the compiled PTX
/// (BuildTime mode) or the raw CUDA source string (Runtime mode) together with the
/// CUDA Driver API dispatch code.
///
/// Runs before CoreCompile. Generated files are added to @(Compile) via the target's
/// Output element so the Roslyn compiler sees them automatically.
/// </summary>
public sealed class CompileCudaKernelsTask : Microsoft.Build.Utilities.Task, ICancelableTask
{
    // ── Cancellation ─────────────────────────────────────────────────────────

    private readonly CancellationTokenSource _cts = new();

    /// <summary>Called by MSBuild on Ctrl+C. Cancels the parallel compilation loop.</summary>
    public void Cancel() => _cts.Cancel();

    // ── Inputs ───────────────────────────────────────────────────────────────

    /// <summary>All C# source files in the project (@(Compile)).</summary>
    [Required]
    public ITaskItem[] CompileItems { get; set; } = [];

    /// <summary>Path to the CUDA include directory (cuda.h, cuda_runtime.h, ...).</summary>
    public string IncludePath { get; set; } = string.Empty;

    /// <summary>
    /// Minimum PTX virtual architecture for build-time compilation.
    /// Accepts any form NVRTC understands: "compute_75", "sm_89", "80".
    /// Defaults to "compute_75".
    /// </summary>
    public string MinArch { get; set; } = "compute_75";

    /// <summary>Extra NVRTC options appended to every kernel in this project (space-separated).</summary>
    public string ExtraOptions { get; set; } = string.Empty;

    /// <summary>Max parallel NVRTC compilations. Empty = all CPU cores.</summary>
    public string MaxParallelism { get; set; } = string.Empty;

    /// <summary>
    /// Verbosity level for KernelSharp messages: "normal" (default) or "detailed".
    /// </summary>
    public string Verbosity { get; set; } = "normal";

    /// <summary>PTX embedding compression: "brotli" (default), "gzip", "zlib", "deflate", or "none".</summary>
    public string PtxCompression { get; set; } = "brotli";

    /// <summary>
    /// Project-wide default compilation mode: "BuildTime" (default) or "Runtime".
    /// Individual kernels may override this with [GpuKernel(Compilation = ...)].
    /// </summary>
    public string Compilation { get; set; } = "BuildTime";

    /// <summary>$(IntermediateOutputPath) — fallback folder for generated .cs files.</summary>
    [Required]
    public string IntermediateOutputPath { get; set; } = string.Empty;

    /// <summary>
    /// Optional committed-source folder for generated .cs files.
    /// When set, files are written here instead of $(IntermediateOutputPath)/KernelSharp/.
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

        // 1. Parse all .cs files to find [GpuKernel] methods
        var kernels = CollectKernels();

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

        // 2. Incremental check
        var toCompile     = new List<KernelInfo>();
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
                    $"KernelSharp:   {k.ClassName}.{k.MethodName} — up-to-date, skipping.");
            }
            else
            {
                toCompile.Add(k);
            }
        }

        // 3. For BuildTime kernels, load NVRTC once before the parallel loop
        bool   nvrtcReady   = false;
        string nvrtcVersion = string.Empty;
        if (toCompile.Any(k => !k.NotImplemented && k.Compilation == KernelCompilationMode.BuildTime))
        {
            try
            {
                NvrtcCompiler.EnsureLoaded();
                nvrtcVersion = NvrtcCompiler.GetVersion();
                nvrtcReady   = true;
                string vSuffix = string.IsNullOrEmpty(nvrtcVersion) ? string.Empty : $" (NVRTC {nvrtcVersion})";
                Log.LogMessage(MessageImportance.Low, $"KernelSharp: NVRTC ready{vSuffix}.");
            }
            catch (Exception ex)
            {
                Log.LogWarning(null, "KERNELSHARP001", null, null, 0, 0, 0, 0,
                    "KernelSharp: NVRTC library not found — BuildTime kernels will have empty PTX " +
                    "and will fail at runtime. " +
                    "Install the CUDA Toolkit and ensure CUDA_PATH is set, " +
                    $"or set KERNELSHARP_CUDA_PATH. ({ex.Message})");
            }
        }

        // 4. Log summary
        int buildTimeCount = toCompile.Count(k => !k.NotImplemented && k.Compilation == KernelCompilationMode.BuildTime);
        int runtimeCount   = toCompile.Count(k => !k.NotImplemented && k.Compilation == KernelCompilationMode.Runtime);
        if (buildTimeCount + runtimeCount > 0)
        {
            var parts = new List<string>();
            if (buildTimeCount > 0) parts.Add($"{buildTimeCount} build-time");
            if (runtimeCount   > 0) parts.Add($"{runtimeCount} runtime");
            string vSuffix = !string.IsNullOrEmpty(nvrtcVersion) ? $" [NVRTC {nvrtcVersion}]" : string.Empty;
            Log.LogMessage(MessageImportance.Normal,
                $"KernelSharp: compiling {string.Join(", ", parts)} CUDA kernel(s){vSuffix}");
        }

        // 5. Compile in parallel (NVRTC is thread-safe per-program-handle)
        int maxPar  = ParseMaxParallelism(MaxParallelism);
        var results = new KernelResult[toCompile.Count];
        System.Threading.Tasks.Parallel.For(0, toCompile.Count,
            new ParallelOptions { MaxDegreeOfParallelism = maxPar },
            i =>
            {
                if (_cts.IsCancellationRequested) return;
                var k = toCompile[i];

                if (k.NotImplemented || k.Compilation == KernelCompilationMode.Runtime)
                {
                    results[i] = new KernelResult(k, null, null);
                    return;
                }

                // BuildTime
                if (!nvrtcReady)
                {
                    results[i] = new KernelResult(k, null, null);
                    return;
                }

                results[i] = CompileWithNvrtc(k, verboseImportance);
            });

        // 6. Write generated .cs files
        bool success = true;
        var generatedPaths = new List<string>(upToDatePaths);

        for (int i = 0; i < results.Length; i++)
        {
            if (results[i] is null) continue;   // cancelled slot

            var r = results[i];

            if (r.Error != null)
            {
                Log.LogError(null, "KERNELSHARP002", null, r.Kernel.SourceFilePath, 0, 0, 0, 0,
                    $"KernelSharp: NVRTC failed for '{r.Kernel.ClassName}.{r.Kernel.MethodName}': {r.Error}");
                success = false;
            }
            else
            {
                string detail = r.Kernel.NotImplemented
                    ? " — stub (NotImplemented)"
                    : r.Kernel.Compilation == KernelCompilationMode.Runtime
                        ? " — runtime (source embedded)"
                    : r.PtxBytes != null
                        ? $" — {r.PtxBytes.Length} bytes PTX"
                    : " — NVRTC skipped (not found)";
                Log.LogMessage(MessageImportance.Normal,
                    $"KernelSharp:   {r.Kernel.ClassName}.{r.Kernel.MethodName}{detail}");
            }

            string genPath = Path.Combine(outDir, $"{r.Kernel.ClassName}.{r.Kernel.MethodName}.g.cs");
            string cs = BuildLauncherSource(r.Kernel, r.PtxBytes,
                EffectiveCompression(r.Kernel.Compression, PtxCompression),
                nvrtcVersion);
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

    // ── NVRTC compilation ─────────────────────────────────────────────────────

    private KernelResult CompileWithNvrtc(KernelInfo k, MessageImportance verboseImportance)
    {
        string arch = string.IsNullOrWhiteSpace(k.Arch) ? MinArch : k.Arch;
        string inc  = !string.IsNullOrWhiteSpace(k.IncludePath) ? k.IncludePath
            : !string.IsNullOrWhiteSpace(IncludePath)          ? IncludePath
            : Environment.GetEnvironmentVariable("CUDA_INCLUDE_PATH") ?? string.Empty;

        string mergedExtra = CombineOptions(ExtraOptions, k.ExtraFlags);

        Log.LogMessage(verboseImportance,
            $"KernelSharp: NVRTC '{k.ClassName}.{k.MethodName}' arch={arch}");

        try
        {
            byte[] ptx = NvrtcCompiler.Compile(k.KernelSource, arch, mergedExtra, inc);
            return new KernelResult(k, ptx, null);
        }
        catch (Exception ex)
        {
            return new KernelResult(k, null, ex.Message);
        }
    }

    private static string CombineOptions(string projectWide, string perKernel)
    {
        string a = (projectWide ?? string.Empty).Trim();
        string b = (perKernel  ?? string.Empty).Trim();
        if (string.IsNullOrEmpty(a)) return b;
        if (string.IsNullOrEmpty(b)) return a;
        return $"{a} {b}";
    }

    // ── Kernel discovery (syntactic Roslyn parsing) ───────────────────────────

    private List<KernelInfo> CollectKernels()
    {
        var result = new List<KernelInfo>();
        foreach (var item in CompileItems)
        {
            string path = item.GetMetadata("FullPath");
            if (string.IsNullOrEmpty(path) || !path.EndsWith(".cs", StringComparison.OrdinalIgnoreCase))
                continue;
            if (!File.Exists(path)) continue;

            string source;
            try { source = File.ReadAllText(path, Encoding.UTF8); }
            catch { continue; }

            if (!source.Contains("GpuKernel", StringComparison.Ordinal)) continue;

            SyntaxTree tree = CSharpSyntaxTree.ParseText(source, new CSharpParseOptions(LanguageVersion.Latest));
            ParseKernels(tree, path, result, ProjectDefaultCompilation());
        }
        return result;
    }

    private KernelCompilationMode ProjectDefaultCompilation() =>
        string.Equals(Compilation, "Runtime", StringComparison.OrdinalIgnoreCase)
            ? KernelCompilationMode.Runtime
            : KernelCompilationMode.BuildTime;

    internal static void ParseKernels(
        SyntaxTree tree, string filePath, List<KernelInfo> kernels,
        KernelCompilationMode projectDefault = KernelCompilationMode.BuildTime)
    {
        var root = tree.GetRoot();
        foreach (var method in root.DescendantNodes().OfType<MethodDeclarationSyntax>())
        {
            if (!method.Modifiers.Any(m => m.IsKind(SyntaxKind.PartialKeyword))) continue;
            if (method.Parent is not ClassDeclarationSyntax cls) continue;
            if (!cls.Modifiers.Any(m => m.IsKind(SyntaxKind.PartialKeyword))) continue;

            AttributeSyntax? gpuAttr = null;
            foreach (var attrList in method.AttributeLists)
            {
                foreach (var attr in attrList.Attributes)
                {
                    string name = attr.Name.ToString();
                    if (name is "GpuKernel" or "GpuKernelAttribute"
                        || name.EndsWith(".GpuKernel",          StringComparison.Ordinal)
                        || name.EndsWith(".GpuKernelAttribute", StringComparison.Ordinal))
                    {
                        gpuAttr = attr;
                        break;
                    }
                }
                if (gpuAttr != null) break;
            }
            if (gpuAttr == null) continue;

            string kernelSource  = string.Empty;
            string sourceFile    = string.Empty;
            string arch          = string.Empty;
            string extraFlags    = string.Empty;
            string incPath       = string.Empty;
            string compression   = string.Empty;
            bool   notImpl       = false;
            int    threadsPerBlock = 0;
            int    blocksPerGrid   = 0;
            KernelCompilationMode compilation = projectDefault;

            var args = gpuAttr.ArgumentList?.Arguments ?? default;
            if (args.Count > 0 && args[0].NameEquals == null)
                kernelSource = ExtractStringLiteral(args[0].Expression);

            foreach (var arg in args)
            {
                if (arg.NameEquals == null) continue;
                switch (arg.NameEquals.Name.Identifier.Text)
                {
                    case "SourceFile":      sourceFile    = ExtractStringLiteral(arg.Expression); break;
                    case "Arch":            arch          = ExtractStringLiteral(arg.Expression); break;
                    case "ExtraFlags":      extraFlags    = ExtractStringLiteral(arg.Expression); break;
                    case "IncludePath":     incPath       = ExtractStringLiteral(arg.Expression); break;
                    case "Compression":     compression   = ExtractStringLiteral(arg.Expression); break;
                    case "NotImplemented":  notImpl       = ExtractBoolLiteral(arg.Expression); break;
                    case "ThreadsPerBlock": threadsPerBlock = ExtractIntLiteral(arg.Expression); break;
                    case "BlocksPerGrid":   blocksPerGrid   = ExtractIntLiteral(arg.Expression); break;
                    case "Compilation":
                        compilation = ExtractEnumMemberName(arg.Expression) == "Runtime"
                            ? KernelCompilationMode.Runtime
                            : KernelCompilationMode.BuildTime;
                        break;
                }
            }

            if (string.IsNullOrWhiteSpace(kernelSource) && !string.IsNullOrWhiteSpace(sourceFile))
            {
                string? dir = Path.GetDirectoryName(filePath);
                string full = !string.IsNullOrEmpty(dir)
                    ? Path.GetFullPath(Path.Combine(dir, sourceFile)) : sourceFile;
                if (File.Exists(full)) kernelSource = File.ReadAllText(full, Encoding.UTF8);
            }

            if (string.IsNullOrWhiteSpace(kernelSource) && !notImpl) continue;

            string ns         = GetNamespace(cls);
            string className  = cls.Identifier.Text;
            string methodName = method.Identifier.Text;
            string paramList  = method.ParameterList.ToString();

            var @params = method.ParameterList.Parameters
                .Select(p => new KernelParam(
                    p.Identifier.Text,
                    p.Type?.ToString() ?? string.Empty,
                    (p.Type?.ToString() ?? string.Empty).StartsWith("CudaBuffer", StringComparison.Ordinal)))
                .ToArray();

            string cudaFuncName = methodName;
            string? validationWarn = null;
            if (!string.IsNullOrEmpty(kernelSource))
            {
                var sig = ExtractCudaSignature(kernelSource);
                if (sig == null)
                {
                    validationWarn =
                        $"No '__global__' function found in kernel source for '{className}.{methodName}'.";
                }
                else
                {
                    cudaFuncName = sig.Value.Name;
                    var warnings = new List<string>();
                    if (!string.Equals(cudaFuncName, methodName, StringComparison.Ordinal))
                        warnings.Add(
                            $"CUDA function name '{cudaFuncName}' does not match C# method name '{methodName}' " +
                            $"in '{className}'. '{cudaFuncName}' will be used for cuModuleGetFunction.");

                    int csCount   = @params.Length;
                    int cudaCount = sig.Value.ParamCount;
                    bool autoN = csCount == cudaCount - 1
                        && @params.Any(p => p.IsBuffer)
                        && !@params.Any(p => !p.IsBuffer && p.TypeSyntax == "int");
                    if (cudaCount != csCount && !autoN)
                        warnings.Add(
                            $"CUDA function '{cudaFuncName}' has {cudaCount} parameter(s) but C# method " +
                            $"'{methodName}' has {csCount}. Verify the signatures match.");

                    if (warnings.Count > 0) validationWarn = string.Join(" ", warnings);
                }
            }

            kernels.Add(new KernelInfo(
                filePath, ns, className, methodName, cudaFuncName,
                kernelSource, arch, extraFlags, incPath,
                paramList, @params, notImpl, compression, compilation,
                threadsPerBlock, blocksPerGrid, validationWarn));
        }
    }

    private static string ExtractStringLiteral(ExpressionSyntax expr) =>
        expr is LiteralExpressionSyntax lit && lit.Token.Value is string s ? s : string.Empty;

    private static bool ExtractBoolLiteral(ExpressionSyntax expr) =>
        expr is LiteralExpressionSyntax lit && lit.Token.Value is bool b && b;

    private static int ExtractIntLiteral(ExpressionSyntax expr) =>
        expr is LiteralExpressionSyntax lit && lit.Token.Value is int i ? i : 0;

    private static string ExtractEnumMemberName(ExpressionSyntax expr)
    {
        if (expr is MemberAccessExpressionSyntax ma) return ma.Name.Identifier.Text;
        if (expr is IdentifierNameSyntax id)         return id.Identifier.Text;
        return string.Empty;
    }

    private static string GetNamespace(ClassDeclarationSyntax cls)
    {
        var parts = new List<string>();
        SyntaxNode? node = cls.Parent;
        while (node != null)
        {
            if (node is NamespaceDeclarationSyntax ns)
                parts.Insert(0, ns.Name.ToString());
            else if (node is FileScopedNamespaceDeclarationSyntax fns)
                parts.Insert(0, fns.Name.ToString());
            node = node.Parent;
        }
        return string.Join(".", parts);
    }

    // ── Incremental build check ───────────────────────────────────────────────

    internal static bool IsUpToDate(string sourceFile, string generatedFile)
    {
        if (!File.Exists(generatedFile)) return false;
        try { return File.GetLastWriteTimeUtc(generatedFile) >= File.GetLastWriteTimeUtc(sourceFile); }
        catch { return false; }
    }

    internal static bool IsStubFile(string generatedFile)
    {
        try
        {
            using var fs = new FileStream(generatedFile, FileMode.Open, FileAccess.Read, FileShare.Read);
            int toRead = (int)Math.Min(fs.Length, 2048);
            byte[] buf = new byte[toRead];
            int read = fs.Read(buf, 0, toRead);
            string header = Encoding.UTF8.GetString(buf, 0, read);
            return header.Contains("NotImplementedException", StringComparison.Ordinal);
        }
        catch { return false; }
    }

    // ── C# launcher source code emission ─────────────────────────────────────

    internal static string BuildLauncherSource(
        KernelInfo k, byte[]? ptxBytes, string compression, string nvrtcVersion = "")
    {
        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine($"// Kernel   : {k.ClassName}.{k.MethodName}");
        sb.AppendLine($"// Mode     : {k.Compilation}");
        if (!string.IsNullOrEmpty(nvrtcVersion))
            sb.AppendLine($"// NVRTC    : {nvrtcVersion}");
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

        if (k.Compilation == KernelCompilationMode.Runtime)
            EmitRuntimeLauncher(sb, k);
        else
            EmitBuildTimeLauncher(sb, k, ptxBytes, compression);

        sb.AppendLine("}");
        return sb.ToString();
    }

    private static void EmitBuildTimeLauncher(
        StringBuilder sb, KernelInfo k, byte[]? ptxBytes, string compression)
    {
        bool compress = compression != "none";
        int  rawLen   = ptxBytes?.Length ?? 0;
        byte[]? embed = rawLen > 0 && compress ? Compress(ptxBytes!, compression) : ptxBytes;
        int  embedLen = embed?.Length ?? 0;

        sb.AppendLine($"    private const string _{k.MethodName}_compression = \"{compression}\";");
        sb.AppendLine();

        if (embedLen > 0)
        {
            sb.AppendLine(compress
                ? $"    // PTX for '{k.MethodName}' — {rawLen} bytes raw, {embedLen} bytes {compression}-compressed"
                : $"    // PTX for '{k.MethodName}' — {rawLen} bytes, embedded uncompressed");
            sb.AppendLine($"    private static readonly byte[] _{k.MethodName}_ptx_encoded =");
            sb.AppendLine("        Convert.FromBase64String(");
            EmitBase64Chunks(sb, Convert.ToBase64String(embed!), "            ");
        }
        else
        {
            sb.AppendLine($"    private static readonly byte[] _{k.MethodName}_ptx_encoded = Array.Empty<byte>();");
        }
        sb.AppendLine();

        sb.AppendLine("#pragma warning disable CS0162 // Unreachable code — compression constant baked in at build time");
        sb.AppendLine($"    private static readonly byte[] _{k.MethodName}_ptx =");
        sb.AppendLine($"        global::KernelSharp.KernelBlobHelper.Decode(_{k.MethodName}_ptx_encoded, _{k.MethodName}_compression);");
        sb.AppendLine("#pragma warning restore CS0162");
        sb.AppendLine();

        AppendModuleAndFuncFields(sb, k);
        sb.AppendLine();

        sb.AppendLine($"    private static void {k.MethodName}_EnsureLoaded()");
        sb.AppendLine("    {");
        sb.AppendLine($"        if (_{k.MethodName}_module != IntPtr.Zero) return;");
        sb.AppendLine("        unsafe");
        sb.AppendLine("        {");
        sb.AppendLine($"            fixed (byte* _p = _{k.MethodName}_ptx)");
        sb.AppendLine("            {");
        sb.AppendLine("                CudaDriverApi.CheckResult(");
        sb.AppendLine($"                    CudaDriverApi.cuModuleLoadData(out _{k.MethodName}_module, (IntPtr)_p));");
        sb.AppendLine("                CudaDriverApi.CheckResult(");
        sb.AppendLine($"                    CudaDriverApi.cuModuleGetFunction(out _{k.MethodName}_func, _{k.MethodName}_module, \"{k.CudaFunctionName}\"));");
        sb.AppendLine("            }");
        sb.AppendLine("        }");
        sb.AppendLine("    }");
        sb.AppendLine();

        AppendDispatchMethod(sb, k);
    }

    private static void EmitRuntimeLauncher(StringBuilder sb, KernelInfo k)
    {
        string escaped = k.KernelSource
            .Replace("\\", "\\\\")
            .Replace("\"", "\\\"")
            .Replace("\r\n", "\\n")
            .Replace("\n", "\\n");

        sb.AppendLine($"    private static readonly string _{k.MethodName}_cudaSource =");
        sb.AppendLine($"        \"{escaped}\";");
        sb.AppendLine();

        AppendModuleAndFuncFields(sb, k);
        sb.AppendLine();

        sb.AppendLine($"    private static void {k.MethodName}_EnsureLoaded()");
        sb.AppendLine("    {");
        sb.AppendLine($"        if (_{k.MethodName}_module != IntPtr.Zero) return;");
        sb.AppendLine($"        string _arch = global::KernelSharp.NvrtcApi.GetNativeArch();");
        sb.AppendLine($"        byte[] _ptx = global::KernelSharp.NvrtcApi.Compile(_{k.MethodName}_cudaSource, _arch);");
        sb.AppendLine("        unsafe");
        sb.AppendLine("        {");
        sb.AppendLine("            fixed (byte* _p = _ptx)");
        sb.AppendLine("            {");
        sb.AppendLine("                CudaDriverApi.CheckResult(");
        sb.AppendLine($"                    CudaDriverApi.cuModuleLoadData(out _{k.MethodName}_module, (IntPtr)_p));");
        sb.AppendLine("                CudaDriverApi.CheckResult(");
        sb.AppendLine($"                    CudaDriverApi.cuModuleGetFunction(out _{k.MethodName}_func, _{k.MethodName}_module, \"{k.CudaFunctionName}\"));");
        sb.AppendLine("            }");
        sb.AppendLine("        }");
        sb.AppendLine("    }");
        sb.AppendLine();

        AppendDispatchMethod(sb, k);
    }

    private static void AppendModuleAndFuncFields(StringBuilder sb, KernelInfo k)
    {
        sb.AppendLine($"    private static IntPtr _{k.MethodName}_module = IntPtr.Zero;");
        sb.AppendLine($"    private static IntPtr _{k.MethodName}_func   = IntPtr.Zero;");
    }

    private static void AppendDispatchMethod(StringBuilder sb, KernelInfo k)
    {
        var bufferParams   = k.Params.Where(p => p.IsBuffer).ToArray();
        var firstIntScalar = k.Params.FirstOrDefault(p => !p.IsBuffer && p.TypeSyntax == "int");
        bool autoInjectN   = firstIntScalar == null && bufferParams.Length > 0;
        int kpSize         = k.Params.Length + (autoInjectN ? 1 : 0);

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
            sb.AppendLine($"            _kp[{i}] = (void*)(&_p{i});");

        string nExpr = firstIntScalar != null ? firstIntScalar.Name
            : bufferParams.Length > 0 ? $"(int){bufferParams[0].Name}.Length" : "1";
        sb.AppendLine($"            int _n = {nExpr};");
        if (autoInjectN)
            sb.AppendLine($"            _kp[{k.Params.Length}] = (void*)(&_n);  // auto-injected n");

        if (k.ThreadsPerBlock > 0 && k.BlocksPerGrid > 0)
        {
            sb.AppendLine($"            uint _threads = {k.ThreadsPerBlock};");
            sb.AppendLine($"            uint _blocks = {k.BlocksPerGrid};");
        }
        else if (k.ThreadsPerBlock > 0)
        {
            sb.AppendLine($"            uint _threads = {k.ThreadsPerBlock};");
            sb.AppendLine("            uint _blocks = (uint)((_n + (int)_threads - 1) / (int)_threads);");
        }
        else if (k.BlocksPerGrid > 0)
        {
            sb.AppendLine("            uint _threads = 256;");
            sb.AppendLine($"            uint _blocks = {k.BlocksPerGrid};");
        }
        else
        {
            sb.AppendLine("            uint _threads = 256;");
            sb.AppendLine("            uint _blocks = (uint)((_n + (int)_threads - 1) / (int)_threads);");
        }

        sb.AppendLine();
        sb.AppendLine($"            CudaDriverApi.CheckResult(CudaDriverApi.cuLaunchKernel(");
        sb.AppendLine($"                _{k.MethodName}_func,");
        sb.AppendLine("                _blocks, 1, 1, _threads, 1, 1,");
        sb.AppendLine("                0, IntPtr.Zero, _kp, null));");
        sb.AppendLine("        }");
        sb.AppendLine("    }");
        sb.AppendLine();
    }

    // ── CUDA signature extraction ─────────────────────────────────────────────

    private static readonly Regex s_globalKernelRx = new(
        @"__global__\s+[\w\s*]+?\s+(\w+)\s*\(([^)]*?)\)",
        RegexOptions.Compiled | RegexOptions.Singleline);

    internal static (string Name, int ParamCount)? ExtractCudaSignature(string source)
    {
        var m = s_globalKernelRx.Match(source);
        if (!m.Success) return null;
        string name       = m.Groups[1].Value;
        string paramBlock = m.Groups[2].Value.Trim();
        int    paramCount = string.IsNullOrEmpty(paramBlock) ? 0 : paramBlock.Split(',').Length;
        return (name, paramCount);
    }

    // ── Code-gen helpers ──────────────────────────────────────────────────────

    internal static void EmitBase64Chunks(StringBuilder sb, string b64, string indent)
    {
        const int chunkSize = 128;
        for (int start = 0; start < b64.Length; start += chunkSize)
        {
            int len  = Math.Min(chunkSize, b64.Length - start);
            string chunk = b64.Substring(start, len);
            bool last = start + len >= b64.Length;
            sb.AppendLine(last ? $"{indent}\"{chunk}\");" : $"{indent}\"{chunk}\" +");
        }
    }

    private static byte[] Compress(byte[] data, string compression)
    {
        var ms = new MemoryStream();
        Stream s = compression switch
        {
            "brotli"  => new BrotliStream(ms, CompressionLevel.Optimal, leaveOpen: true),
            "gzip"    => new GZipStream(ms, CompressionLevel.Optimal, leaveOpen: true),
            "zlib"    => new ZLibStream(ms, CompressionLevel.Optimal, leaveOpen: true),
            "deflate" => new DeflateStream(ms, CompressionLevel.Optimal, leaveOpen: true),
            _         => throw new ArgumentException($"Unknown compression format: {compression}"),
        };
        using (s)
            s.Write(data, 0, data.Length);
        return ms.ToArray();
    }

    internal static string EffectiveCompression(string attrOverride, string projectDefault) =>
        string.IsNullOrWhiteSpace(attrOverride)
            ? projectDefault
            : attrOverride.Trim().ToLowerInvariant();

    internal static int ParseMaxParallelism(string value) =>
        !string.IsNullOrWhiteSpace(value)
        && int.TryParse(value.Trim(), out int parsed) && parsed > 0
            ? parsed : -1;

    // ── Data models ───────────────────────────────────────────────────────────

    internal enum KernelCompilationMode { BuildTime, Runtime }

    internal sealed class KernelInfo(
        string sourceFilePath, string ns, string className, string methodName, string cudaFunctionName,
        string kernelSource, string arch, string extraFlags, string incPath,
        string paramList, KernelParam[] @params, bool notImpl, string compression,
        KernelCompilationMode compilation,
        int threadsPerBlock = 0, int blocksPerGrid = 0,
        string? validationWarning = null)
    {
        public string SourceFilePath     { get; } = sourceFilePath;
        public string Namespace          { get; } = ns;
        public string ClassName          { get; } = className;
        public string MethodName         { get; } = methodName;
        public string CudaFunctionName   { get; } = cudaFunctionName;
        public string KernelSource       { get; } = kernelSource;
        public string Arch               { get; } = arch;
        public string ExtraFlags         { get; } = extraFlags;
        public string IncludePath        { get; } = incPath;
        public string ParameterList      { get; } = paramList;
        public KernelParam[] Params      { get; } = @params;
        public bool NotImplemented       { get; } = notImpl;
        public string Compression        { get; } = compression;
        public KernelCompilationMode Compilation { get; } = compilation;
        public int ThreadsPerBlock       { get; } = threadsPerBlock;
        public int BlocksPerGrid         { get; } = blocksPerGrid;
        public string? ValidationWarning { get; } = validationWarning;
    }

    internal sealed class KernelParam(string name, string typeSyntax, bool isBuffer)
    {
        public string Name       { get; } = name;
        public string TypeSyntax { get; } = typeSyntax;
        public bool   IsBuffer   { get; } = isBuffer;
    }

    private sealed class KernelResult(KernelInfo kernel, byte[]? ptxBytes, string? error)
    {
        public KernelInfo Kernel   { get; } = kernel;
        public byte[]?    PtxBytes { get; } = ptxBytes;
        public string?    Error    { get; } = error;
    }
}
