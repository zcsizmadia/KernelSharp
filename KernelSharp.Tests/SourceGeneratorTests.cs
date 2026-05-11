// Tests for the MSBuild task's code generation logic – no GPU needed.
// Exercises: attribute recognition, fatbin field generation, namespace handling,
// partial method wiring, compression, multi-kernel output.

using System.Collections.Generic;
using System.Linq;

using KernelSharp.Build;

using Microsoft.CodeAnalysis.CSharp;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace KernelSharp.Tests;

public class SourceGeneratorTests
{
    // ── Helper ────────────────────────────────────────────────────────────────

    /// <summary>
    /// Parse <paramref name="source"/>, discover [GpuKernel] methods via the task's
    /// syntactic Roslyn walk, and emit C# launcher source for each one.
    /// No GPU or nvcc required.
    /// </summary>
    private static List<string> GenerateSources(string source, string projectCompression = "gzip")
    {
        var tree = CSharpSyntaxTree.ParseText(source,
            new CSharpParseOptions(LanguageVersion.Latest));
        var kernels = new List<CompileCudaKernelsTask.KernelInfo>();
        CompileCudaKernelsTask.ParseKernels(tree, string.Empty, kernels);
        return kernels
            .Select(k => CompileCudaKernelsTask.BuildLauncherSource(
                k, null,
                CompileCudaKernelsTask.EffectiveCompression(k.Compression, projectCompression),
                CompileCudaKernelsTask.CompilerInfo.Empty,
                string.Empty))
            .ToList();
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
        var files = GenerateSources(src);

        await Assert.That(files.Count).IsGreaterThan(0)
            .Because("the task must emit at least one file");
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
        var files = GenerateSources(src);
        string generated = string.Concat(files);

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
        var files = GenerateSources(src);
        string generated = string.Concat(files);

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
        var files = GenerateSources(src);
        string generated = string.Concat(files);

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
        var files = GenerateSources(src);
        string generated = string.Concat(files);

        await Assert.That(generated).Contains("MyFancyKernel");
    }

    [Test]
    public async Task Generator_DoesNotEmit_ForNonPartialClass()
    {
        const string src = """
            using KernelSharp;
            namespace Ns {
                // Not marked partial – task must skip this
                public class NotPartial {
                    [GpuKernel("__global__ void K() {}", NotImplemented = true)]
                    public void K(CudaBuffer<float> b) { }
                }
            }
            """;
        var files = GenerateSources(src);
        // Nothing should be generated for a non-partial class
        await Assert.That(files.Count).IsEqualTo(0);
    }

    [Test]
    public async Task Generator_DoesNotEmit_ForNonPartialMethod()
    {
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class C {
                // Not marked partial – task must skip this method
                [GpuKernel("__global__ void K() {}", NotImplemented = true)]
                public void K(CudaBuffer<float> b) { }
            }}
            """;
        var files = GenerateSources(src);
        await Assert.That(files.Count).IsEqualTo(0)
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
        var files = GenerateSources(src);
        await Assert.That(files.Count).IsGreaterThanOrEqualTo(2);
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
        var files = GenerateSources(src);
        string generated = string.Concat(files);

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
        var files = GenerateSources(src);
        string generated = string.Concat(files);

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
        var files = GenerateSources(src);
        string generated = string.Concat(files);

        await Assert.That(generated).Contains("_MyKernel_compression");
    }

    [Test]
    public async Task Generator_EmitsCompressionGzip_ByDefault()
    {
        // When no Compression attribute arg is provided, the project default ("gzip") applies.
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class K {
                [GpuKernel("extern \"C\" __global__ void NoCompKernel(float* b, int n) {}")]
                public partial void NoCompKernel(CudaBuffer<float> b);
            }}
            """;
        var files = GenerateSources(src);
        string generated = string.Concat(files);

        await Assert.That(generated).Contains("_NoCompKernel_compression = \"gzip\"");
    }

    [Test]
    public async Task Generator_EmitsCompressionNone_WhenAttributeSpecifiesNone()
    {
        // When Compression = "none" is set on the attribute, the generated constant
        // must reflect that so the loader does not apply decompression.
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class K {
                [GpuKernel("extern \"C\" __global__ void NoCompKernel(float* b, int n) {}", Compression = "none")]
                public partial void NoCompKernel(CudaBuffer<float> b);
            }}
            """;
        var files = GenerateSources(src);
        string generated = string.Concat(files);

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
        var files = GenerateSources(src);
        string generated = string.Concat(files);

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
        var files = GenerateSources(src);
        string generated = string.Concat(files);

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
        var files = GenerateSources(src);
        string generated = string.Concat(files);

        await Assert.That(generated).Contains("_MyKernel_fatbin = _MyKernel_DecodeFatbin()");
    }

    [Test]
    public async Task Generator_CompressionOverride_DoesNotAffectOtherKernels()
    {
        // When two kernels share the same class but have different Compression values,
        // each must carry its own independent compression constant.
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class M {
                [GpuKernel("extern \"C\" __global__ void KernelA(float* b, int n) {}", Compression = "none")]
                public partial void KernelA(CudaBuffer<float> b);
                [GpuKernel("extern \"C\" __global__ void KernelB(float* b, int n) {}", Compression = "gzip")]
                public partial void KernelB(CudaBuffer<float> b);
            }}
            """;
        var files = GenerateSources(src);

        // Each kernel gets its own file — check them individually
        string? fileA = files.FirstOrDefault(s => s.Contains("_KernelA_compression"));
        string? fileB = files.FirstOrDefault(s => s.Contains("_KernelB_compression"));

        await Assert.That(fileA).IsNotNull();
        await Assert.That(fileB).IsNotNull();
        await Assert.That(fileA!).Contains("_KernelA_compression = \"none\"");
        await Assert.That(fileB!).Contains("_KernelB_compression = \"gzip\"");
    }
}