// Tests for the source generator itself – no GPU needed.
// Exercises: attribute recognition, fatbin field generation, namespace handling,
// partial method wiring, diagnostics, nvcc path discovery.

using System.Reflection;

using KernelSharp.SourceGenerator;

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace KernelSharp.Tests;

public class SourceGeneratorTests
{
    // ── Helpers ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Run the generator against <paramref name="source"/> and return all
    /// generated syntax trees (excludes the input).
    /// </summary>
    private static (IReadOnlyList<Diagnostic> Diagnostics,
                    IReadOnlyList<SyntaxTree> GeneratedTrees)
        RunGenerator(string source)
    {
        var parseOptions = CSharpParseOptions.Default
            .WithLanguageVersion(LanguageVersion.Latest);

        var compilation = CSharpCompilation.Create(
            "TestAssembly",
            [CSharpSyntaxTree.ParseText(source, parseOptions)],
            GetMetadataReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));

        var generator = new GpuKernelGenerator();
        var driver = CSharpGeneratorDriver
            .Create(new IIncrementalGenerator[] { generator })
            .RunGeneratorsAndUpdateCompilation(compilation,
                out _, out var diagnostics);

        // GetRunResult() is the canonical way to retrieve generator output
        var result = driver.GetRunResult();
        var generated = result.GeneratedTrees;

        return (diagnostics, generated);
    }

    private static IEnumerable<MetadataReference> GetMetadataReferences()
    {
        // Include every trusted platform assembly loaded in this process so that
        // Roslyn can resolve CudaBuffer<T>, IDisposable, Span<T>, etc.
        // This is the canonical approach for in-process Roslyn generator tests on .NET 6+.
        var trustedPaths = (AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") as string)
            ?? string.Empty;
        foreach (string path in trustedPaths.Split(';', StringSplitOptions.RemoveEmptyEntries))
            yield return MetadataReference.CreateFromFile(path);

        // KernelSharp runtime (GpuKernelAttribute, CudaBuffer<T>, …)
        yield return MetadataReference.CreateFromFile(
            typeof(GpuKernelAttribute).Assembly.Location);
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    [Test]
    public async Task Generator_ProducesOutput_ForValidKernel()
    {
        // NotImplemented = true skips nvcc but still emits a partial-method stub
        const string src = """
            using KernelSharp;
            namespace MyNs
            {
                public partial class MyKernels
                {
                    [GpuKernel("__global__ void AddKernel(float* a, float* b, float* c, int n) {}", NotImplemented = true)]
                    public partial void AddKernel(CudaBuffer<float> a, CudaBuffer<float> b, CudaBuffer<float> c);
                }
            }
            """;
        var (_, trees) = RunGenerator(src);

        await Assert.That(trees.Count).IsGreaterThan(0)
            .Because("the generator must emit at least one file");
    }

    [Test]
    public async Task Generator_EmitsFatbinConstant_InGeneratedFile()
    {
        // Regular kernel (no NotImplemented) — fatbin boilerplate is always emitted
        // even when nvcc is absent (empty byte array + full CUDA driver setup code).
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class K {
                [GpuKernel("extern \"C\" __global__ void MyKernel(float* b, int n) {}")]
                public partial void MyKernel(CudaBuffer<float> b);
            }}
            """;
        var (_, trees) = RunGenerator(src);
        string generated = string.Concat(trees.Select(t => t.ToString()));

        // The generated code must contain the fatbin field and the EnsureLoaded helper
        await Assert.That(generated).Contains("_MyKernel_fatbin");
        await Assert.That(generated).Contains("MyKernel_EnsureLoaded");
    }

    [Test]
    public async Task Generator_EmitsCuModuleLoadData_Call()
    {
        // Regular kernel — full Driver API boilerplate is always emitted (fatbin may be empty
        // when nvcc is absent, but the cuModuleLoadData / cuLaunchKernel calls are always present).
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class K {
                [GpuKernel("extern \"C\" __global__ void MyKernel(float* b, int n) {}")]
                public partial void MyKernel(CudaBuffer<float> b);
            }}
            """;
        var (_, trees) = RunGenerator(src);
        string generated = string.Concat(trees.Select(t => t.ToString()));

        await Assert.That(generated).Contains("cuModuleLoadData");
        await Assert.That(generated).Contains("cuModuleGetFunction");
        await Assert.That(generated).Contains("cuLaunchKernel");
    }

    [Test]
    public async Task Generator_UsesCorrectNamespace_InOutput()
    {
        const string src = """
            using KernelSharp;
            namespace Deep.Nested.Ns { public partial class Foo {
                [GpuKernel("__global__ void Bar(float* x, int n) {}", NotImplemented = true)]
                public partial void Bar(CudaBuffer<float> x);
            }}
            """;
        var (_, trees) = RunGenerator(src);
        string generated = string.Concat(trees.Select(t => t.ToString()));

        await Assert.That(generated).Contains("namespace Deep.Nested.Ns");
    }

    [Test]
    public async Task Generator_DerivesKernelFunctionName_FromMethodName()
    {
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class C {
                [GpuKernel("__global__ void MyFancyKernel(float* b, int n) {}", NotImplemented = true)]
                public partial void MyFancyKernel(CudaBuffer<float> b);
            }}
            """;
        var (_, trees) = RunGenerator(src);
        string generated = string.Concat(trees.Select(t => t.ToString()));

        await Assert.That(generated).Contains("MyFancyKernel");
    }

    [Test]
    public async Task Generator_DoesNotEmit_ForNonPartialClass()
    {
        const string src = """
            using KernelSharp;
            namespace Ns {
                // Not marked partial – generator must skip this
                public class NotPartial {
                    [GpuKernel("__global__ void K() {}", NotImplemented = true)]
                    public void K(CudaBuffer<float> b) { }
                }
            }
            """;
        var (_, trees) = RunGenerator(src);
        // Nothing should be generated for a non-partial class
        await Assert.That(trees.Count).IsEqualTo(0);
    }

    [Test]
    public async Task Generator_DoesNotEmit_ForNonPartialMethod()
    {
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class C {
                // Not marked partial – generator must skip this method
                [GpuKernel("__global__ void K() {}", NotImplemented = true)]
                public void K(CudaBuffer<float> b) { }
            }}
            """;
        var (_, trees) = RunGenerator(src);
        await Assert.That(trees.Count).IsEqualTo(0)
            .Because("non-partial methods are not kernel launch sites");
    }

    [Test]
    public async Task Generator_GeneratesMultipleFiles_ForMultipleMethods()
    {
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class Multi {
                [GpuKernel("__global__ void K1(float* a, int n) {}", NotImplemented = true)]
                public partial void K1(CudaBuffer<float> a);
                [GpuKernel("__global__ void K2(float* b, int n) {}", NotImplemented = true)]
                public partial void K2(CudaBuffer<float> b);
            }}
            """;
        var (_, trees) = RunGenerator(src);
        await Assert.That(trees.Count).IsGreaterThanOrEqualTo(2);
    }

    [Test]
    public async Task Generator_EmitsPartialMethodImplementation_WithCorrectParams()
    {
        // Regular kernel — DevicePointer extraction is always emitted for CudaBuffer<T> params.
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class C {
                [GpuKernel("extern \"C\" __global__ void Add(float* a, float* b, float* c, int n) {}")]
                public partial void Add(CudaBuffer<float> a, CudaBuffer<float> b, CudaBuffer<float> c);
            }}
            """;
        var (_, trees) = RunGenerator(src);
        string generated = string.Concat(trees.Select(t => t.ToString()));

        // The generated partial method must extract DevicePointer for each CudaBuffer param
        await Assert.That(generated).Contains("a.DevicePointer");
        await Assert.That(generated).Contains("b.DevicePointer");
        await Assert.That(generated).Contains("c.DevicePointer");
    }

    [Test]
    public async Task Generator_FileScopedNamespace_IsHandledCorrectly()
    {
        const string src = """
            using KernelSharp;
            namespace Ns;

            public partial class C {
                [GpuKernel("__global__ void K(float* b, int n) {}", NotImplemented = true)]
                public partial void K(CudaBuffer<float> b);
            }
            """;
        var (_, trees) = RunGenerator(src);
        string generated = string.Concat(trees.Select(t => t.ToString()));

        await Assert.That(generated).Contains("namespace Ns");
    }

    // ── Compression tests ─────────────────────────────────────────────────────

    [Test]
    public async Task Generator_EmitsCompressionConstant_InGeneratedFile()
    {
        // The generated code must always contain the self-describing compression constant
        // so that the embedded blob can always be decoded correctly.
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class K {
                [GpuKernel("extern \"C\" __global__ void MyKernel(float* b, int n) {}")]
                public partial void MyKernel(CudaBuffer<float> b);
            }}
            """;
        var (_, trees) = RunGenerator(src);
        string generated = string.Concat(trees.Select(t => t.ToString()));

        await Assert.That(generated).Contains("_MyKernel_compression");
    }

    [Test]
    public async Task Generator_EmitsCompressionNone_ByDefault()
    {
        // When no Compression attribute arg is provided, the project default applies.
        // In the test environment there's no MSBuild context, so the default is "none".
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class K {
                [GpuKernel("extern \"C\" __global__ void NoCompKernel(float* b, int n) {}")]
                public partial void NoCompKernel(CudaBuffer<float> b);
            }}
            """;
        var (_, trees) = RunGenerator(src);
        string generated = string.Concat(trees.Select(t => t.ToString()));

        await Assert.That(generated).Contains("_NoCompKernel_compression = \"none\"");
    }

    [Test]
    public async Task Generator_EmitsCompressionGzip_WhenAttributeSpecifiesGzip()
    {
        // When Compression = "gzip" is set on the attribute, the generated constant
        // must reflect that so the loader applies decompression.
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class K {
                [GpuKernel("extern \"C\" __global__ void GzipKernel(float* b, int n) {}", Compression = "gzip")]
                public partial void GzipKernel(CudaBuffer<float> b);
            }}
            """;
        var (_, trees) = RunGenerator(src);
        string generated = string.Concat(trees.Select(t => t.ToString()));

        await Assert.That(generated).Contains("_GzipKernel_compression = \"gzip\"");
    }

    [Test]
    public async Task Generator_EmitsDecodeFatbin_InGeneratedFile()
    {
        // The generated code must always contain the unified decode helper
        // so that the fatbin is decoded using the stored compression constant.
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class K {
                [GpuKernel("extern \"C\" __global__ void MyKernel(float* b, int n) {}")]
                public partial void MyKernel(CudaBuffer<float> b);
            }}
            """;
        var (_, trees) = RunGenerator(src);
        string generated = string.Concat(trees.Select(t => t.ToString()));

        await Assert.That(generated).Contains("_MyKernel_DecodeFatbin");
        await Assert.That(generated).Contains("_MyKernel_fatbin_encoded");
    }

    [Test]
    public async Task Generator_EmitsFatbinField_ThatCallsDecodeFatbin()
    {
        // The final _fatbin field must be initialised by calling _DecodeFatbin(),
        // making the decode path deterministic and the generated file self-contained.
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class K {
                [GpuKernel("extern \"C\" __global__ void MyKernel(float* b, int n) {}")]
                public partial void MyKernel(CudaBuffer<float> b);
            }}
            """;
        var (_, trees) = RunGenerator(src);
        string generated = string.Concat(trees.Select(t => t.ToString()));

        await Assert.That(generated).Contains("_MyKernel_fatbin = _MyKernel_DecodeFatbin()");
    }

    [Test]
    public async Task Generator_CompressionOverride_DoesNotAffectOtherKernels()
    {
        // When two kernels share the same class but only one has Compression = "gzip",
        // each must carry its own independent compression constant.
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class M {
                [GpuKernel("extern \"C\" __global__ void KernelA(float* b, int n) {}")]
                public partial void KernelA(CudaBuffer<float> b);
                [GpuKernel("extern \"C\" __global__ void KernelB(float* b, int n) {}", Compression = "gzip")]
                public partial void KernelB(CudaBuffer<float> b);
            }}
            """;
        var (_, trees) = RunGenerator(src);

        // Each kernel gets its own file — check them individually
        string? fileA = trees.Select(t => t.ToString()).FirstOrDefault(s => s.Contains("_KernelA_compression"));
        string? fileB = trees.Select(t => t.ToString()).FirstOrDefault(s => s.Contains("_KernelB_compression"));

        await Assert.That(fileA).IsNotNull();
        await Assert.That(fileB).IsNotNull();
        await Assert.That(fileA!).Contains("_KernelA_compression = \"none\"");
        await Assert.That(fileB!).Contains("_KernelB_compression = \"gzip\"");
    }
}