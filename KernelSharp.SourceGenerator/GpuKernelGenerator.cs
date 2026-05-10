using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace KernelSharp.SourceGenerator;

[Generator(LanguageNames.CSharp)]
public sealed class GpuKernelGenerator : IIncrementalGenerator
{
    private const string AttributeFullName = "KernelSharp.GpuKernelAttribute";

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        // The attribute now lives on the partial METHOD declaration.
        var kernelMethods = context.SyntaxProvider
            .ForAttributeWithMetadataName(
                AttributeFullName,
                predicate: static (node, _) => node is MethodDeclarationSyntax,
                transform: static (ctx, ct) => TransformKernelMethod(ctx, ct))
            .Where(static x => x is not null)
            .Select(static (x, _) => x!);

        var buildProps = context.AnalyzerConfigOptionsProvider
            .Select(static (opts, _) =>
            {
                opts.GlobalOptions.TryGetValue("build_property.KernelSharpIncludePath", out string? inc);
                opts.GlobalOptions.TryGetValue("build_property.KernelSharpNvccStd", out string? std);
                opts.GlobalOptions.TryGetValue("build_property.KernelSharpCCCLPath", out string? ccclPath);
                opts.GlobalOptions.TryGetValue("build_property.KernelSharpNvccExtraFlags", out string? nvccExtra);
                opts.GlobalOptions.TryGetValue("build_property.KernelSharpMsvcClPath", out string? clPath);
                opts.GlobalOptions.TryGetValue("build_property.KernelSharpTargetArchs", out string? targetArchs);
                opts.GlobalOptions.TryGetValue("build_property.KernelSharpMaxParallelism", out string? maxPar);
                opts.GlobalOptions.TryGetValue("build_property.KernelSharpFatbinCompression", out string? fatbinComp);
                int maxParallelism = -1; // -1 = use all available cores (Parallel.For default)
                if (maxPar is not null && int.TryParse(maxPar.Trim(), out int parsedPar) && parsedPar > 0)
                    maxParallelism = parsedPar;
                return new BuildConfig(
                    includePath: inc ?? string.Empty,
                    nvccStd: std ?? "c++20",
                    ccclPath: ccclPath ?? string.Empty,
                    nvccExtraFlags: nvccExtra ?? string.Empty,
                    msvcClPath: clPath ?? string.Empty,
                    targetArchs: targetArchs ?? "compute_80",
                    maxParallelism: maxParallelism,
                    fatbinCompression: (fatbinComp ?? "none").Trim().ToLowerInvariant());
            });

        var batch = kernelMethods.Collect().Combine(buildProps);
        context.RegisterSourceOutput(batch,
            (ctx, pair) => GenerateAllKernels(ctx, pair.Left, pair.Right));
    }

    // ── Transform ────────────────────────────────────────────────────────────────────────

    private static KernelMethodModel? TransformKernelMethod(
        GeneratorAttributeSyntaxContext ctx,
        System.Threading.CancellationToken ct)
    {
        if (ctx.TargetNode is not MethodDeclarationSyntax method) return null;
        if (method.Parent is not ClassDeclarationSyntax classSyntax) return null;
        if (!classSyntax.Modifiers.Any(m => m.IsKind(SyntaxKind.PartialKeyword))) return null;
        if (!method.Modifiers.Any(m => m.IsKind(SyntaxKind.PartialKeyword))) return null;

        AttributeData attr = ctx.Attributes[0];

        // --- Resolve kernel source ---
        string kernelSource = string.Empty;

        // 1. Positional constructor arg: [GpuKernel("...")]  or  [GpuKernel("""...""")]
        if (attr.ConstructorArguments.Length > 0)
            kernelSource = attr.ConstructorArguments[0].Value?.ToString() ?? string.Empty;

        // 2. SourceFile property: [GpuKernel(SourceFile = "path/to/kernel.cu")]
        string sourceFile = GetNamedArg(attr, "SourceFile", string.Empty);
        if (string.IsNullOrWhiteSpace(kernelSource) && !string.IsNullOrWhiteSpace(sourceFile))
        {
            // Resolve relative to the source file that contains the attribute
            string? dir = Path.GetDirectoryName(method.SyntaxTree.FilePath);
            string fullPath = !string.IsNullOrEmpty(dir)
                ? Path.GetFullPath(Path.Combine(dir, sourceFile))
                : sourceFile;
            if (File.Exists(fullPath))
                kernelSource = File.ReadAllText(fullPath, Encoding.UTF8);
        }

        if (string.IsNullOrWhiteSpace(kernelSource)) return null;

        string arch = GetNamedArg(attr, "Arch", string.Empty);
        string extraFlags = GetNamedArg(attr, "ExtraFlags", string.Empty);
        string incPath = GetNamedArg(attr, "IncludePath", string.Empty);
        string compression = GetNamedArg(attr, "Compression", string.Empty);
        bool notImpl = GetNamedBoolArg(attr, "NotImplemented", false);

        string methodName = method.Identifier.Text;
        string paramList = method.ParameterList.ToString();

        var paramInfos = method.ParameterList.Parameters
            .Select(p =>
            {
                string paramName = p.Identifier.Text;
                string typeSyntax = p.Type!.ToString();
                var typeSymbol = ctx.SemanticModel.GetTypeInfo(p.Type!, ct).Type;
                // CudaBuffer<T> and CudaBuffer both have Name == "CudaBuffer"
                bool isBuffer = typeSymbol?.Name == "CudaBuffer";
                return new KernelParamInfo(paramName, typeSyntax,
                    isBuffer ? KernelParamKind.Buffer : KernelParamKind.Scalar);
            })
            .ToArray();

        string ns = GetNamespace(classSyntax);
        string className = classSyntax.Identifier.Text;

        return new KernelMethodModel(ns, className, methodName, kernelSource,
            arch, extraFlags, incPath, paramList, paramInfos, notImpl, compression);
    }

    // ── Code generation ──────────────────────────────────────────────────────────────────

    private static void GenerateAllKernels(
        SourceProductionContext ctx,
        ImmutableArray<KernelMethodModel> models,
        BuildConfig cfg)
    {
        if (models.IsEmpty) return;

        // Resolve nvcc and cl.exe once — these scan PATH/env/vswhere, skip per-kernel cost
        string nvcc = FindNvcc();
        bool isWin = OperatingSystem_IsWindows();
        string clDir = isWin ? FindClDir(cfg.MsvcClPath) : string.Empty;

        if (string.IsNullOrEmpty(nvcc))
        {
            foreach (var m in models)
            {
                ctx.ReportDiagnostic(Diagnostic.Create(Diagnostics.NvccNotFound, Location.None, m.MethodName));
                string eff = EffectiveCompression(m.Compression, cfg.FatbinCompression);
                ctx.AddSource($"{m.ClassName}.{m.MethodName}.g.cs",
                    SourceText.From(BuildLauncherSource(m, null, eff, CompilerInfo.Query(string.Empty, string.Empty), string.Empty), Encoding.UTF8));
            }
            return;
        }

        // Pre-compute per-kernel inputs sequentially (cheap)
        var workItems = new (KernelMethodModel model, string effectiveInc, string[] archs)[models.Length];
        for (int i = 0; i < models.Length; i++)
        {
            var model = models[i];
            string effectiveInc = !string.IsNullOrWhiteSpace(model.IncludePath)
                ? model.IncludePath
                : !string.IsNullOrWhiteSpace(cfg.IncludePath)
                    ? cfg.IncludePath
                    : Environment.GetEnvironmentVariable("CUDA_INCLUDE_PATH") ?? string.Empty;

            string archOverride = string.IsNullOrWhiteSpace(model.Arch) ? string.Empty : NormalizeArch(model.Arch);
            string[] archs = string.IsNullOrEmpty(archOverride)
                ? cfg.TargetArchs
                    .Split(new[] { ';', ',', ' ' }, StringSplitOptions.RemoveEmptyEntries)
                    .Select(NormalizeArch).Distinct().ToArray()
                : new[] { archOverride };
            if (archs.Length == 0) archs = new[] { "compute_80" };

            workItems[i] = (model, effectiveInc, archs);
        }

        // Query tool versions once — cheap one-shot processes, done before the parallel loop
        CompilerInfo compInfo = CompilerInfo.Query(nvcc, clDir);

        // Compile all kernels in parallel — each spawns one nvcc process.
        // NotImplemented kernels skip nvcc entirely.
        // MaxDegreeOfParallelism: -1 = use all cores; set KernelSharpMaxParallelism to limit
        var results = new (byte[]? fatbin, string nvccArgs, Diagnostic? diagnostic)[workItems.Length];
        Parallel.For(0, workItems.Length,
            new ParallelOptions { MaxDegreeOfParallelism = cfg.MaxParallelism },
            i =>
            {
                if (workItems[i].model.NotImplemented) { results[i] = (null, string.Empty, null); return; }
                var (model, effectiveInc, archs) = workItems[i];
                results[i] = CompileToFatbin(model, cfg, effectiveInc, archs, nvcc, clDir);
            });

        // Emit all sources on the Roslyn thread — SourceProductionContext is not thread-safe
        for (int i = 0; i < workItems.Length; i++)
        {
            var model = workItems[i].model;
            var (fatbin, nvccArgs, diagnostic) = results[i];
            string eff = EffectiveCompression(model.Compression, cfg.FatbinCompression);
            if (diagnostic != null) ctx.ReportDiagnostic(diagnostic);
            ctx.AddSource($"{model.ClassName}.{model.MethodName}.g.cs",
                SourceText.From(BuildLauncherSource(model, fatbin, eff, compInfo, nvccArgs), Encoding.UTF8));
        }
    }

    /// <summary>Returns the effective compression: attribute override if set, else project default.</summary>
    private static string EffectiveCompression(string attrOverride, string projectDefault) =>
        string.IsNullOrWhiteSpace(attrOverride)
            ? projectDefault
            : attrOverride.Trim().ToLowerInvariant();

    private static string BuildLauncherSource(
        KernelMethodModel model, byte[]? fatbin, string compression,
        CompilerInfo compInfo, string nvccArgs)
    {
        var sb = new StringBuilder();
        sb.AppendLine("// <auto-generated/>");
        sb.AppendLine($"// Kernel   : {model.ClassName}.{model.MethodName}");
        if (!string.IsNullOrEmpty(compInfo.CudaVersion)) sb.AppendLine($"// CUDA     : {compInfo.CudaVersion}");
        if (!string.IsNullOrEmpty(compInfo.NvccVersion)) sb.AppendLine($"// nvcc     : {compInfo.NvccVersion}");
        if (!string.IsNullOrEmpty(compInfo.HostCompiler)) sb.AppendLine($"// Compiler : {compInfo.HostCompiler}");
        if (!string.IsNullOrEmpty(nvccArgs)) sb.AppendLine($"// nvcc args: {nvccArgs}");
        sb.AppendLine("#nullable enable");
        sb.AppendLine("using System;");
        sb.AppendLine("using KernelSharp;");
        sb.AppendLine();

        if (!string.IsNullOrEmpty(model.Namespace))
        {
            sb.AppendLine($"namespace {model.Namespace};");
            sb.AppendLine();
        }

        sb.AppendLine($"partial class {model.ClassName}");
        sb.AppendLine("{");

        // NotImplemented stub: no fatbin, no CUDA fields, just throw
        if (model.NotImplemented)
        {
            sb.AppendLine($"    public partial void {model.MethodName}{model.ParameterList}");
            sb.AppendLine("        => throw new NotImplementedException(");
            sb.AppendLine($"            $\"{model.ClassName}.{model.MethodName} is marked [GpuKernel(NotImplemented=true)]\");");
            sb.AppendLine();
            sb.AppendLine("}");
            return sb.ToString();
        }

        // Embed fatbin bytes (optionally gzip-compressed) as a base64 string literal.
        // The compression format is stored as a const so the generated file is self-describing:
        // readers and the decode helper can always determine how to interpret the blob.
        bool useGzip = compression == "gzip";
        int rawLen = fatbin?.Length ?? 0;
        byte[]? embedBytes = (rawLen > 0 && useGzip) ? GZipCompress(fatbin!) : fatbin;
        int embedLen = embedBytes?.Length ?? 0;

        // Always emit the compression constant — makes the generated file self-describing
        sb.AppendLine($"    // Compression format used when embedding the fatbin at build time.");
        sb.AppendLine($"    private const string _{model.MethodName}_compression = \"{compression}\";");
        sb.AppendLine();

        // Emit the (possibly compressed) blob
        if (embedLen > 0)
        {
            string blobComment = useGzip
                ? $"    // fatbin for '{model.MethodName}' — {rawLen} bytes raw, {embedLen} bytes gzip-compressed, embedded at build time"
                : $"    // fatbin for '{model.MethodName}' — {rawLen} bytes, embedded uncompressed at build time";
            sb.AppendLine(blobComment);
            sb.AppendLine($"    private static readonly byte[] _{model.MethodName}_fatbin_encoded =");
            sb.AppendLine("        Convert.FromBase64String(");
            EmitBase64Chunks(sb, Convert.ToBase64String(embedBytes!), "            ");
        }
        else
        {
            sb.AppendLine($"    private static readonly byte[] _{model.MethodName}_fatbin_encoded = Array.Empty<byte>();");
        }
        sb.AppendLine();

        // Unified decode helper — branches on the compression constant baked in at build time.
        // The branch taken is always the same; the constant makes the generated code self-documenting.
        sb.AppendLine($"#pragma warning disable CS0162 // Unreachable code — intentional: compression constant is baked in at build time");
        sb.AppendLine($"    private static byte[] _{model.MethodName}_DecodeFatbin()");
        sb.AppendLine("    {");
        sb.AppendLine($"        if (_{model.MethodName}_compression == \"none\")");
        sb.AppendLine($"            return _{model.MethodName}_fatbin_encoded;");
        sb.AppendLine($"        using (var _ms = new global::System.IO.MemoryStream(_{model.MethodName}_fatbin_encoded))");
        sb.AppendLine("        using (var _gz = new global::System.IO.Compression.GZipStream(");
        sb.AppendLine("            _ms, global::System.IO.Compression.CompressionMode.Decompress))");
        sb.AppendLine("        {");
        sb.AppendLine("            var _out = new global::System.IO.MemoryStream();");
        sb.AppendLine("            _gz.CopyTo(_out);");
        sb.AppendLine("            return _out.ToArray();");
        sb.AppendLine("        }");
        sb.AppendLine("    }");
        sb.AppendLine($"#pragma warning restore CS0162");
        sb.AppendLine();
        sb.AppendLine($"    private static readonly byte[] _{model.MethodName}_fatbin = _{model.MethodName}_DecodeFatbin();");
        sb.AppendLine();
        sb.AppendLine($"    private static IntPtr _{model.MethodName}_module = IntPtr.Zero;");
        sb.AppendLine($"    private static IntPtr _{model.MethodName}_func   = IntPtr.Zero;");
        sb.AppendLine();

        // EnsureLoaded: pass fatbin bytes directly to CUDA driver — driver selects best SASS arch
        sb.AppendLine($"    private static void {model.MethodName}_EnsureLoaded()");
        sb.AppendLine("    {");
        sb.AppendLine($"        if (_{model.MethodName}_module != IntPtr.Zero) return;");
        sb.AppendLine($"        unsafe");
        sb.AppendLine($"        {{");
        sb.AppendLine($"            fixed (byte* _p = _{model.MethodName}_fatbin)");
        sb.AppendLine($"            {{");
        sb.AppendLine($"                CudaDriverApi.CheckResult(");
        sb.AppendLine($"                    CudaDriverApi.cuModuleLoadData(");
        sb.AppendLine($"                        out _{model.MethodName}_module, (IntPtr)_p));");
        sb.AppendLine($"                CudaDriverApi.CheckResult(");
        sb.AppendLine($"                    CudaDriverApi.cuModuleGetFunction(");
        sb.AppendLine($"                        out _{model.MethodName}_func,");
        sb.AppendLine($"                        _{model.MethodName}_module,");
        sb.AppendLine($"                        \"{model.MethodName}\"));");
        sb.AppendLine($"            }}");
        sb.AppendLine($"        }}");
        sb.AppendLine("    }");
        sb.AppendLine();

        // Partial method implementation
        var bufferParams = model.Params.Where(p => p.IsBuffer).ToArray();
        var firstIntScalar = model.Params.FirstOrDefault(p => !p.IsBuffer && p.TypeSyntax == "int");
        // Auto-inject n when no int scalar param is present (kernel expects it as last CUDA arg)
        bool autoInjectN = firstIntScalar == null && bufferParams.Length > 0;
        int kpSize = model.Params.Length + (autoInjectN ? 1 : 0);

        sb.AppendLine($"    public partial void {model.MethodName}{model.ParameterList}");
        sb.AppendLine("    {");
        sb.AppendLine($"        {model.MethodName}_EnsureLoaded();");
        sb.AppendLine("        unsafe");
        sb.AppendLine("        {");
        for (int i = 0; i < model.Params.Length; i++)
        {
            var p = model.Params[i];
            if (p.IsBuffer)
                sb.AppendLine($"            IntPtr _p{i} = {p.Name}.DevicePointer;");
            else
                sb.AppendLine($"            {p.TypeSyntax} _p{i} = {p.Name};");
        }
        sb.AppendLine($"            void** _kp = stackalloc void*[{kpSize}];");
        for (int i = 0; i < model.Params.Length; i++)
            sb.AppendLine($"            _kp[{i}] = (void*)(&_p{i});");

        string nExpr = firstIntScalar != null
            ? firstIntScalar.Name
            : bufferParams.Length > 0 ? $"(int){bufferParams[0].Name}.Length" : "1";
        sb.AppendLine($"            int _n = {nExpr};");
        if (autoInjectN)
            sb.AppendLine($"            _kp[{model.Params.Length}] = (void*)(&_n);  // auto-injected n");
        sb.AppendLine("            uint _threads = 256;");
        sb.AppendLine("            uint _blocks = (uint)((_n + (int)_threads - 1) / (int)_threads);");
        sb.AppendLine();
        sb.AppendLine($"            CudaDriverApi.CheckResult(CudaDriverApi.cuLaunchKernel(");
        sb.AppendLine($"                _{model.MethodName}_func,");
        sb.AppendLine("                _blocks, 1, 1, _threads, 1, 1,");
        sb.AppendLine("                0, IntPtr.Zero, _kp, null));");
        sb.AppendLine("        }");
        sb.AppendLine("    }");
        sb.AppendLine();
        sb.AppendLine("}");
        return sb.ToString();
    }

    /// <summary>Emits base64 string chunks as a concatenated C# string expression ending with a semicolon.</summary>
    private static void EmitBase64Chunks(StringBuilder sb, string b64, string indent)
    {
        const int chunkSize = 128;
        for (int start = 0; start < b64.Length; start += chunkSize)
        {
            int len = Math.Min(chunkSize, b64.Length - start);
            string chunk = b64.Substring(start, len);
            bool last = start + len >= b64.Length;
            sb.AppendLine(last
                ? $"{indent}\"{chunk}\");"
                : $"{indent}\"{chunk}\" +");
        }
    }

    /// <summary>GZip-compresses <paramref name="data"/> using optimal compression level.</summary>
    private static byte[] GZipCompress(byte[] data)
    {
        var ms = new MemoryStream();
        using (var gz = new GZipStream(ms, CompressionLevel.Optimal, leaveOpen: true))
            gz.Write(data, 0, data.Length);
        return ms.ToArray();
    }

    // ── nvcc invocation ──────────────────────────────────────────────────────────────────

    // Thread-safe: compiles kernel to fatbin in a temp directory, returns the raw bytes
    // and the nvcc args string (without the -ccbin prefix, for embedding in the comment).
    private static (byte[]? fatbin, string nvccArgs, Diagnostic? diagnostic) CompileToFatbin(
        KernelMethodModel model, BuildConfig cfg,
        string effectiveInc, string[] archs,
        string nvcc, string clDir)
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "KernelSharpGen");
        Directory.CreateDirectory(tempDir);
        string srcFile = Path.Combine(tempDir, $"{model.MethodName}.cu");
        string fatbinFile = Path.Combine(tempDir, $"{model.MethodName}.fatbin");

        File.WriteAllText(srcFile, model.KernelSource, Encoding.UTF8);

        bool isWin = OperatingSystem_IsWindows();
        char sep = isWin ? '\\' : '/';
        var args = new StringBuilder();

        // -ccbin is prepended first but excluded from the recorded args (it's just a host path)
        string ccbinPrefix = (isWin && !string.IsNullOrEmpty(clDir))
            ? $"-ccbin \"{clDir}\" "
            : string.Empty;
        if (!string.IsNullOrEmpty(ccbinPrefix)) args.Append(ccbinPrefix);

        // Multi-arch fatbinary: one output file, compiled SASS for each arch.
        // cuModuleLoadData loads the fatbin directly — driver picks the best SASS match.
        // Include a virtual-arch (PTX) section for the lowest arch as a forward-compat fallback.
        args.Append("-fatbin ");
        string? lowestArch = null;
        foreach (string arch in archs)
        {
            string num = arch.Replace("compute_", "");
            // Compiled SASS for this arch
            args.Append($"-gencode arch={arch},code=sm_{num} ");
            if (lowestArch == null) lowestArch = arch; // first = lowest (archs ordered in config)
        }
        // Virtual-arch (PTX) section — future GPU generations JIT-compile from this
        if (lowestArch != null)
            args.Append($"-gencode arch={lowestArch},code={lowestArch} ");
        string std = string.IsNullOrWhiteSpace(cfg.NvccStd) ? "c++20" : cfg.NvccStd;
        args.Append($"-x cu -std={std} --extended-lambda --use_fast_math ");

        string cudaRoot = GetCudaRoot(nvcc);
        if (!string.IsNullOrWhiteSpace(effectiveInc))
        {
            string inc = effectiveInc.TrimEnd('\\', '/');
            args.Append($"-I\"{inc}\" ");
            string cccl = ResolveCcclPath(cfg.CcclPath, inc, cudaRoot, isWin);
            if (!string.IsNullOrEmpty(cccl))
            {
                args.Append($"-I\"{cccl}{sep}thrust\" ");
                args.Append($"-I\"{cccl}{sep}libcudacxx{sep}include\" ");
                args.Append($"-I\"{cccl}{sep}cub\" ");
            }
        }
        if (!string.IsNullOrEmpty(cudaRoot))
        {
            args.Append($"-I\"{cudaRoot}{sep}include\" ");
            string cccl13 = Path.Combine(cudaRoot, "include", "cccl");
            if (Directory.Exists(cccl13)) args.Append($"-I\"{cccl13}\" ");
        }

        args.Append("-D_USE_MATH_DEFINES ");
        if (isWin)
        {
            args.Append("-D_WINDOWS -DFMT_UNICODE=0 -DNDEBUG -DNOMINMAX -DWIN32_LEAN_AND_MEAN -DNOGDI ");
            args.Append("-Xcompiler \"/EHsc /Zc:__cplusplus /utf-8 /wd4996 /wd4100 /wd4864 /wd4702 /wd4324 /wd4714 /Zc:preprocessor /WX\" ");
        }
        else
            args.Append("-Xcompiler \"-fPIC\" ");

        if (!string.IsNullOrWhiteSpace(cfg.NvccExtraFlags)) args.Append($"{cfg.NvccExtraFlags} ");
        if (!string.IsNullOrWhiteSpace(model.ExtraFlags)) args.Append($"{model.ExtraFlags} ");

        // Record clean args (no -ccbin path, no src/output paths) for the generated comment
        string cleanArgs = args.ToString()
            .Replace(ccbinPrefix, string.Empty)
            .Trim();

        args.Append($"\"{srcFile}\" -o \"{fatbinFile}\"");

        var psi = new ProcessStartInfo
        {
            FileName = nvcc,
            Arguments = args.ToString(),
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        var proc = new Process { StartInfo = psi };
        if (!proc.Start())
            return (null, string.Empty, Diagnostic.Create(Diagnostics.NvccFailed, Location.None,
                model.MethodName, "Could not start nvcc"));

        string stderr = proc.StandardError.ReadToEnd();
        string stdout = proc.StandardOutput.ReadToEnd();
        proc.WaitForExit();
        int exitCode = proc.ExitCode;
        proc.Dispose();

        if (exitCode != 0)
        {
            string combined = (stderr + stdout).Trim();
            return (null, string.Empty, Diagnostic.Create(Diagnostics.NvccFailed, Location.None,
                model.MethodName, combined));
        }

        return (File.Exists(fatbinFile) ? File.ReadAllBytes(fatbinFile) : null, cleanArgs, null);
    }

    // ── Compiler version info ─────────────────────────────────────────────────────────────

    private sealed class CompilerInfo
    {
        public string CudaVersion { get; }   // e.g. "13.2"
        public string NvccVersion { get; }   // e.g. "V13.2.78"
        public string HostCompiler { get; }   // e.g. "MSVC 19.41.36231" or "gcc 14.2.0"

        private CompilerInfo(string cuda, string nvcc, string host)
        {
            CudaVersion = cuda;
            NvccVersion = nvcc;
            HostCompiler = host;
        }

        /// <summary>Runs nvcc --version and (cl.exe or gcc --version) to collect build metadata.</summary>
        internal static CompilerInfo Query(string nvcc, string clDir)
        {
            string cudaVer = string.Empty;
            string nvccVer = string.Empty;
            string hostVer = string.Empty;

            // nvcc --version — output looks like:
            //   nvcc: NVIDIA (R) Cuda compiler driver
            //   ...
            //   Cuda compilation tools, release 13.2, V13.2.78
            if (!string.IsNullOrEmpty(nvcc))
            {
                string nvccOut = RunProcess(nvcc, "--version", useStderr: false);
                foreach (string line in nvccOut.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries))
                {
                    // "Cuda compilation tools, release 13.2, V13.2.78"
                    int ri = line.IndexOf("release ", StringComparison.OrdinalIgnoreCase);
                    int vi = line.IndexOf(", V", StringComparison.OrdinalIgnoreCase);
                    if (ri >= 0 && vi > ri)
                    {
                        cudaVer = line.Substring(ri + 8, vi - ri - 8).Trim();
                        nvccVer = line.Substring(vi + 3).Trim();
                        break;
                    }
                }
            }

            bool isWin = OperatingSystem_IsWindows();
            if (isWin && !string.IsNullOrEmpty(clDir))
            {
                // cl.exe prints its banner to stderr when invoked with no arguments
                string clExe = Path.Combine(clDir, "cl.exe");
                if (File.Exists(clExe))
                {
                    // First line: "Microsoft (R) C/C++ Optimizing Compiler Version 19.41.36231 for x64"
                    string clOut = RunProcess(clExe, string.Empty, useStderr: true);
                    string firstLine = clOut.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                                           .FirstOrDefault() ?? string.Empty;
                    // Compress to "MSVC 19.41.36231 x64"
                    int verIdx = firstLine.IndexOf("Version ", StringComparison.OrdinalIgnoreCase);
                    if (verIdx >= 0)
                    {
                        string rest = firstLine.Substring(verIdx + 8).Trim(); // "19.41.36231 for x64"
                        rest = rest.Replace(" for ", " ").Trim();
                        hostVer = "MSVC " + rest;
                    }
                    else if (firstLine.Length > 0)
                        hostVer = "MSVC " + firstLine;
                }
            }
            else if (!isWin)
            {
                // gcc --version: "gcc (Ubuntu 14.2.0-4ubuntu2) 14.2.0" or similar
                string gccOut = RunProcess("gcc", "--version", useStderr: false);
                string firstLine = gccOut.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                                         .FirstOrDefault() ?? string.Empty;
                if (firstLine.Length > 0)
                {
                    // Extract the trailing version number "14.2.0"
                    string[] parts = firstLine.Split(' ');
                    string ver = parts.Length > 0 ? parts[parts.Length - 1].Trim() : firstLine;
                    hostVer = "gcc " + ver;
                }
            }

            return new CompilerInfo(cudaVer, nvccVer, hostVer);
        }

        private static string RunProcess(string exe, string arguments, bool useStderr)
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = exe,
                    Arguments = arguments,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                };
                var p = new Process { StartInfo = psi };
                if (!p.Start()) return string.Empty;
                string stdout = p.StandardOutput.ReadToEnd();
                string stderr = p.StandardError.ReadToEnd();
                p.WaitForExit();
                p.Dispose();
                return useStderr ? stderr : stdout;
            }
            catch { return string.Empty; }
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────────────────

    private static string NormalizeArch(string arch)
    {
        arch = arch.Trim();
        if (arch.StartsWith("compute_")) return arch;
        if (arch.StartsWith("sm_")) return "compute_" + arch.Substring(3);
        if (arch.Contains('.')) return "compute_" + arch.Replace(".", "");
        return "compute_" + arch;
    }

    private static string ResolveCcclPath(string explicitCcclPath, string inc, string cudaRoot, bool isWin)
    {
        if (!string.IsNullOrWhiteSpace(explicitCcclPath) && Directory.Exists(explicitCcclPath))
            return explicitCcclPath.TrimEnd('\\', '/');
        if (!string.IsNullOrWhiteSpace(inc))
        {
            string matxRoot = Path.GetDirectoryName(inc.TrimEnd('\\', '/')) ?? string.Empty;
            string fromBuild = Path.Combine(matxRoot, "build", "_deps", "cccl-src");
            if (Directory.Exists(fromBuild)) return fromBuild;
        }
        if (!string.IsNullOrEmpty(cudaRoot))
        {
            string bundled = Path.Combine(cudaRoot, "include", "cccl");
            if (Directory.Exists(bundled)) return bundled;
        }
        return string.Empty;
    }

    private static string GetCudaRoot(string nvcc)
    {
        string nvccDir = Path.GetDirectoryName(nvcc) ?? string.Empty;
        return Path.GetDirectoryName(nvccDir) ?? string.Empty;
    }

    private static string FindClDir(string explicitClPath)
    {
        if (!string.IsNullOrWhiteSpace(explicitClPath))
        {
            string c = explicitClPath.Trim('"').TrimEnd('\\', '/');
            if (File.Exists(c)) return Path.GetDirectoryName(c) ?? string.Empty;
            if (Directory.Exists(c)) return c;
        }
        string? vcTools = Environment.GetEnvironmentVariable("VCToolsInstallDir");
        if (!string.IsNullOrEmpty(vcTools))
        {
            string dir = Path.Combine(vcTools.TrimEnd('\\', '/'), "bin", "HostX64", "x64");
            if (Directory.Exists(dir)) return dir;
        }
        string pf86 = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86);
        string vswhere = Path.Combine(pf86, "Microsoft Visual Studio", "Installer", "vswhere.exe");
        if (File.Exists(vswhere)) { string? d = TryFindClViaVswhere(vswhere); if (d != null) return d; }
        string? d2 = ScanVsRootsForCl();
        if (d2 != null) return d2;
        string? pathEnv = Environment.GetEnvironmentVariable("PATH");
        if (pathEnv != null)
            foreach (string entry in pathEnv.Split(Path.PathSeparator))
                if (File.Exists(Path.Combine(entry.Trim(), "cl.exe"))) return entry.Trim();
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
            foreach (string line in output.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries))
            {
                string? d = FindClInVsRoot(line.Trim());
                if (d != null) return d;
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
            if (!Directory.Exists(vsRoot)) continue;
            string[] majors = Directory.GetDirectories(vsRoot);
            Array.Sort(majors, StringComparer.OrdinalIgnoreCase);
            Array.Reverse(majors);
            foreach (string major in majors)
            {
                string[] eds = Directory.GetDirectories(major);
                Array.Sort(eds, StringComparer.OrdinalIgnoreCase);
                Array.Reverse(eds);
                foreach (string ed in eds) { string? d = FindClInVsRoot(ed); if (d != null) return d; }
            }
        }
        return null;
    }

    private static string? FindClInVsRoot(string vsRoot)
    {
        string msvcRoot = Path.Combine(vsRoot, "VC", "Tools", "MSVC");
        if (!Directory.Exists(msvcRoot)) return null;
        string[] versions = Directory.GetDirectories(msvcRoot);
        Array.Sort(versions, StringComparer.OrdinalIgnoreCase);
        Array.Reverse(versions);
        foreach (string ver in versions)
        {
            string dir = Path.Combine(ver, "bin", "HostX64", "x64");
            if (File.Exists(Path.Combine(dir, "cl.exe"))) return dir;
        }
        return null;
    }

    private static string FindNvcc()
    {
        bool isWin = OperatingSystem_IsWindows();
        string nvccExe = isWin ? "nvcc.exe" : "nvcc";

        foreach (string ev in new[] { "CUDA_PATH", "CUDA_TOOLKIT_ROOT_DIR" })
        {
            string? root = Environment.GetEnvironmentVariable(ev);
            if (!string.IsNullOrEmpty(root))
            {
                string c = Path.Combine(root, "bin", nvccExe);
                if (File.Exists(c)) return c;
            }
        }

        string? pathEnv = Environment.GetEnvironmentVariable("PATH");
        if (pathEnv != null)
            foreach (string d in pathEnv.Split(Path.PathSeparator))
            { string c = Path.Combine(d.Trim(), nvccExe); if (File.Exists(c)) return c; }

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
                { string c = Path.Combine(ver, "bin", nvccExe); if (File.Exists(c)) return c; }
            }
        }
        else
        {
            foreach (string prefix in new[] { "/usr/local/cuda/bin", "/usr/bin" })
            { string c = Path.Combine(prefix, nvccExe); if (File.Exists(c)) return c; }
        }

        return string.Empty;
    }

    private static bool OperatingSystem_IsWindows() =>
        System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
            System.Runtime.InteropServices.OSPlatform.Windows);

    private static string GetNamespace(ClassDeclarationSyntax cls)
    {
        SyntaxNode? parent = cls.Parent;
        var parts = new List<string>();
        while (parent is not null)
        {
            if (parent is NamespaceDeclarationSyntax n)
                parts.Insert(0, n.Name.ToString());
            else if (parent is FileScopedNamespaceDeclarationSyntax f)
                parts.Insert(0, f.Name.ToString());
            parent = parent.Parent;
        }
        return string.Join(".", parts);
    }

    private static string GetNamedArg(AttributeData attr, string name, string def)
    {
        foreach (var kv in attr.NamedArguments)
            if (kv.Key == name) return kv.Value.Value?.ToString() ?? def;
        return def;
    }

    private static bool GetNamedBoolArg(AttributeData attr, string name, bool def)
    {
        foreach (var kv in attr.NamedArguments)
            if (kv.Key == name) return kv.Value.Value is bool b ? b : def;
        return def;
    }
}