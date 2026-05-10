using Microsoft.CodeAnalysis;

namespace KernelSharp.SourceGenerator;

internal static class Diagnostics
{
    public static readonly DiagnosticDescriptor NvccNotFound = new(
        id: "MATXGEN001",
        title: "nvcc not found",
        messageFormat: "Could not locate nvcc for kernel field '{0}'. " +
                       "Set CUDA_TOOLKIT_ROOT_DIR or ensure nvcc is on PATH.",
        category: "KernelSharp.SourceGenerator",
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    public static readonly DiagnosticDescriptor NvccFailed = new(
        id: "MATXGEN002",
        title: "nvcc compilation failed",
        messageFormat: "nvcc failed for kernel field '{0}': {1}",
        category: "KernelSharp.SourceGenerator",
        // Warning – a failed nvcc just means an empty fatbin; the C# partial launcher
        // is still emitted so the project compiles. The failure surfaces at runtime.
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    public static readonly DiagnosticDescriptor KernelMethodMismatch = new(
        id: "MATXGEN003",
        title: "Kernel launcher method not found",
        messageFormat: "No partial method found in '{0}' to pair with kernel field '{1}'. " +
                       "Declare a partial method in the same class.",
        category: "KernelSharp.SourceGenerator",
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true);
}