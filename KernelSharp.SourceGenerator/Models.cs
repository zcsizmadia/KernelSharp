// netstandard2.0 does not have IsExternalInit – use plain sealed classes.
namespace KernelSharp.SourceGenerator;

internal enum KernelParamKind { Buffer, Scalar }

/// <summary>Describes a single parameter of a [GpuKernel] partial method.</summary>
internal sealed class KernelParamInfo
{
    /// <summary>C# parameter name (e.g. "a", "eps", "n").</summary>
    public string Name { get; }
    /// <summary>Type as written in source (e.g. "CudaBuffer&lt;float&gt;", "int", "float").</summary>
    public string TypeSyntax { get; }
    public KernelParamKind Kind { get; }
    public bool IsBuffer => Kind == KernelParamKind.Buffer;

    public KernelParamInfo(string name, string typeSyntax, KernelParamKind kind)
    {
        Name = name;
        TypeSyntax = typeSyntax;
        Kind = kind;
    }
}

/// <summary>
/// Represents a single [GpuKernel]-decorated partial method and its CUDA source.
/// The C# method name is also the CUDA extern "C" kernel function name.
/// </summary>
internal sealed class KernelMethodModel
{
    public string Namespace { get; }
    public string ClassName { get; }
    /// <summary>C# method name AND the CUDA function name (must match).</summary>
    public string MethodName { get; }
    public string KernelSource { get; }
    public string Arch { get; }
    public string ExtraFlags { get; }
    public string IncludePath { get; }
    public string ParameterList { get; }   // "(CudaBuffer<float> a, int n)" — verbatim syntax
    public KernelParamInfo[] Params { get; }
    /// <summary>When true, skip nvcc and generate throw new NotImplementedException().</summary>
    public bool NotImplemented { get; }
    /// <summary>Per-kernel compression override. Empty string means "use project setting".</summary>
    public string Compression { get; }

    public KernelMethodModel(
        string @namespace,
        string className,
        string methodName,
        string kernelSource,
        string arch,
        string extraFlags,
        string includePath,
        string parameterList,
        KernelParamInfo[] @params,
        bool notImplemented = false,
        string compression = "")
    {
        Namespace = @namespace;
        ClassName = className;
        MethodName = methodName;
        KernelSource = kernelSource;
        Arch = arch;
        ExtraFlags = extraFlags;
        IncludePath = includePath;
        ParameterList = parameterList;
        Params = @params;
        NotImplemented = notImplemented;
        Compression = compression;
    }
}

/// <summary>
/// Project-wide nvcc build settings forwarded from MSBuild via CompilerVisibleProperty.
/// All properties are individually overridable in a .csproj or Directory.Build.props.
/// </summary>
internal sealed class BuildConfig
{
    public string IncludePath { get; }
    public string NvccStd { get; }
    public string CcclPath { get; }
    public string NvccExtraFlags { get; }
    public string MsvcClPath { get; }
    /// <summary>Semicolon-separated virtual arch list, e.g. "compute_80;compute_89;compute_90".</summary>
    public string TargetArchs { get; }
    /// <summary>Maximum parallel nvcc processes. -1 = use all available cores.</summary>
    public int MaxParallelism { get; }
    /// <summary>Compression applied to the embedded fatbin bytes. "none" (default) or "gzip".</summary>
    public string FatbinCompression { get; }

    public BuildConfig(
        string includePath,
        string nvccStd,
        string ccclPath,
        string nvccExtraFlags,
        string msvcClPath,
        string targetArchs,
        int maxParallelism = -1,
        string fatbinCompression = "none")
    {
        IncludePath = includePath;
        NvccStd = nvccStd;
        CcclPath = ccclPath;
        NvccExtraFlags = nvccExtraFlags;
        MsvcClPath = msvcClPath;
        TargetArchs = targetArchs;
        MaxParallelism = maxParallelism;
        FatbinCompression = fatbinCompression;
    }
}