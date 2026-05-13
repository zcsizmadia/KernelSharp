// Tests for the MSBuild task's code generation logic – no GPU needed.
// Exercises: attribute recognition, ptx field generation, namespace handling,
// partial method wiring, compression, multi-kernel output, Runtime mode.

using System.Collections.Generic;
using System.IO;
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
    /// No GPU or NVRTC required.
    /// </summary>
    private static List<string> GenerateSources(string source, string projectCompression = "brotli")
    {
        var tree = CSharpSyntaxTree.ParseText(source,
            new CSharpParseOptions(LanguageVersion.Latest));
        var kernels = new List<CompileCudaKernelsTask.KernelInfo>();
        CompileCudaKernelsTask.ParseKernels(tree, string.Empty, kernels);
        return [.. kernels
            .Select(k => CompileCudaKernelsTask.BuildLauncherSource(
                k, null,
                CompileCudaKernelsTask.EffectiveCompression(k.Compression, projectCompression)))];
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    [Test]
    public async Task Generator_ProducesOutput_ForValidKernel()
    {
        // NotImplemented = true skips NVRTC but still emits a partial-method stub
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
    public async Task Generator_EmitsPtxConstant_InGeneratedFile()
    {
        // Regular kernel — PTX boilerplate is always emitted even when NVRTC is absent
        // (empty byte array + full CUDA driver setup code).
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class K {
                [GpuKernel("extern \"C\" __global__ void MyKernel(float* b, int n) {}")]
                public partial void MyKernel(CudaBuffer<float> b);
            }}
            """;
        var files = GenerateSources(src);
        string generated = string.Concat(files);

        // The generated code must contain the ptx field and the EnsureLoaded helper
        await Assert.That(generated).Contains("_MyKernel_ptx");
        await Assert.That(generated).Contains("MyKernel_EnsureLoaded");
    }

    [Test]
    public async Task Generator_EmitsCuModuleLoadData_Call()
    {
        // Regular kernel — full Driver API boilerplate is always emitted (ptx may be empty
        // when NVRTC is absent, but the cuModuleLoadData / cuLaunchKernel calls are always present).
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
    public async Task Generator_EmitsCompressionBrotli_ByDefault()
    {
        // When no Compression attribute arg is provided, the project default ("brotli") applies.
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class K {
                [GpuKernel("extern \"C\" __global__ void NoCompKernel(float* b, int n) {}")]
                public partial void NoCompKernel(CudaBuffer<float> b);
            }}
            """;
        var files = GenerateSources(src);
        string generated = string.Concat(files);

        await Assert.That(generated).Contains("_NoCompKernel_compression = \"brotli\"");
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
    public async Task Generator_EmitsDecodeKernelBlob_InGeneratedFile()
    {
        // The generated code must use KernelBlobHelper.Decode so the ptx is
        // decoded via the shared helper rather than per-kernel inline code.
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class K {
                [GpuKernel("extern \"C\" __global__ void MyKernel(float* b, int n) {}")]
                public partial void MyKernel(CudaBuffer<float> b);
            }}
            """;
        var files = GenerateSources(src);
        string generated = string.Concat(files);

        await Assert.That(generated).Contains("KernelBlobHelper.Decode");
        await Assert.That(generated).Contains("_MyKernel_ptx_encoded");
    }

    [Test]
    public async Task Generator_EmitsPtxField_ThatCallsDecodeKernelBlob()
    {
        // The _ptx field must be initialised by KernelBlobHelper.Decode so the
        // decode path uses the shared helper and passes the compression constant.
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class K {
                [GpuKernel("extern \"C\" __global__ void MyKernel(float* b, int n) {}")]
                public partial void MyKernel(CudaBuffer<float> b);
            }}
            """;
        var files = GenerateSources(src);
        string generated = string.Concat(files);

        await Assert.That(generated).Contains("global::KernelSharp.KernelBlobHelper.Decode(_MyKernel_ptx_encoded, _MyKernel_compression)");
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

    // ── Runtime mode tests ────────────────────────────────────────────────────

    [Test]
    public async Task Generator_RuntimeMode_EmitsCudaSourceField()
    {
        // Runtime mode must embed the CUDA source string instead of PTX bytes.
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class K {
                [GpuKernel("extern \"C\" __global__ void RtKernel(float* b, int n) {}",
                    Compilation = KernelCompilation.Runtime)]
                public partial void RtKernel(CudaBuffer<float> b);
            }}
            """;
        var tree = CSharpSyntaxTree.ParseText(src, new CSharpParseOptions(LanguageVersion.Latest));
        var kernels = new List<CompileCudaKernelsTask.KernelInfo>();
        CompileCudaKernelsTask.ParseKernels(tree, string.Empty, kernels);
        string generated = CompileCudaKernelsTask.BuildLauncherSource(kernels[0], null, "gzip");

        await Assert.That(generated).Contains("_RtKernel_cudaSource");
        await Assert.That(generated).DoesNotContain("_RtKernel_ptx_encoded")
            .Because("Runtime mode embeds source, not pre-compiled PTX");
    }

    [Test]
    public async Task Generator_RuntimeMode_EmitsNvrtcApiCalls()
    {
        // Runtime mode must call NvrtcApi.GetNativeArch() and NvrtcApi.Compile().
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class K {
                [GpuKernel("extern \"C\" __global__ void RtKernel(float* b, int n) {}",
                    Compilation = KernelCompilation.Runtime)]
                public partial void RtKernel(CudaBuffer<float> b);
            }}
            """;
        var tree = CSharpSyntaxTree.ParseText(src, new CSharpParseOptions(LanguageVersion.Latest));
        var kernels = new List<CompileCudaKernelsTask.KernelInfo>();
        CompileCudaKernelsTask.ParseKernels(tree, string.Empty, kernels);
        string generated = CompileCudaKernelsTask.BuildLauncherSource(kernels[0], null, "gzip");

        await Assert.That(generated).Contains("NvrtcApi.GetNativeArch");
        await Assert.That(generated).Contains("NvrtcApi.Compile");
    }

    [Test]
    public async Task Generator_RuntimeMode_DoesNotEmitCompressionConstant()
    {
        // Runtime mode does not compress/embed PTX, so no compression constant.
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class K {
                [GpuKernel("extern \"C\" __global__ void RtKernel(float* b, int n) {}",
                    Compilation = KernelCompilation.Runtime)]
                public partial void RtKernel(CudaBuffer<float> b);
            }}
            """;
        var tree = CSharpSyntaxTree.ParseText(src, new CSharpParseOptions(LanguageVersion.Latest));
        var kernels = new List<CompileCudaKernelsTask.KernelInfo>();
        CompileCudaKernelsTask.ParseKernels(tree, string.Empty, kernels);
        string generated = CompileCudaKernelsTask.BuildLauncherSource(kernels[0], null, "gzip");

        await Assert.That(generated).DoesNotContain("_RtKernel_compression")
            .Because("Runtime mode has no embedded PTX to decompress");
    }

    // ── SourceFile tests ──────────────────────────────────────────────────────

    [Test]
    public async Task Generator_ParseKernels_LoadsSourceFromExternalCuFile()
    {
        string tempPath = Path.Combine(Path.GetTempPath(), $"ks_test_{System.Guid.NewGuid():N}.cu");
        try
        {
            // Write a real .cu file with a recognisable function name.
            const string cuSource =
                "extern \"C\" __global__ void FromFileKernel(float* a, int n) { }";
            System.IO.File.WriteAllText(tempPath, cuSource, System.Text.Encoding.UTF8);

            // Build C# source that references the file by absolute path.
            string src = $$"""
                using KernelSharp;
                namespace Ns;
                public partial class C {
                    [GpuKernel(SourceFile = @"{{tempPath}}")]
                    public partial void FromFileKernel(CudaBuffer<float> a);
                }
                """;

            var tree = CSharpSyntaxTree.ParseText(src, new CSharpParseOptions(LanguageVersion.Latest));
            var kernels = new List<CompileCudaKernelsTask.KernelInfo>();
            CompileCudaKernelsTask.ParseKernels(tree, string.Empty, kernels);

            await Assert.That(kernels.Count).IsEqualTo(1)
                .Because("one [GpuKernel(SourceFile=…)] method should be discovered");
            await Assert.That(kernels[0].KernelSource).Contains("FromFileKernel")
                .Because("KernelSource must be populated from the .cu file contents");
        }
        finally
        {
            if (System.IO.File.Exists(tempPath))
            {
                System.IO.File.Delete(tempPath);
            }
        }
    }

    // ── NvrtcApi.NormalizeArchOption ──────────────────────────────────────────

    [Test]
    public async Task NormalizeArchOption_AlreadyPrefixed_ReturnsUnchanged()
    {
        await Assert.That(NvrtcCompiler.NormalizeArchOption("compute_89"))
            .IsEqualTo("--gpu-architecture=compute_89");
    }

    [Test]
    public async Task NormalizeArchOption_SmPrefix_ReturnsSm()
    {
        await Assert.That(NvrtcCompiler.NormalizeArchOption("sm_80"))
            .IsEqualTo("--gpu-architecture=sm_80");
    }

    [Test]
    public async Task NormalizeArchOption_DottedVersion_ConvertsToPrefixed()
    {
        await Assert.That(NvrtcCompiler.NormalizeArchOption("8.9"))
            .IsEqualTo("--gpu-architecture=compute_89");
    }

    [Test]
    public async Task NormalizeArchOption_PlainNumber_ConvertsToPrefixed()
    {
        await Assert.That(NvrtcCompiler.NormalizeArchOption("90"))
            .IsEqualTo("--gpu-architecture=compute_90");
    }

    [Test]
    public async Task NormalizeArchOption_FullFlag_ReturnsUnchanged()
    {
        const string flag = "--gpu-architecture=compute_75";
        await Assert.That(NvrtcCompiler.NormalizeArchOption(flag)).IsEqualTo(flag);
    }

    // ── ParseMaxParallelism ───────────────────────────────────────────────────

    [Test]
    public async Task ParseMaxParallelism_ValidPositive_ReturnsValue()
    {
        await Assert.That(CompileCudaKernelsTask.ParseMaxParallelism("4")).IsEqualTo(4);
    }

    [Test]
    public async Task ParseMaxParallelism_EmptyString_ReturnsMinusOne()
    {
        await Assert.That(CompileCudaKernelsTask.ParseMaxParallelism(string.Empty)).IsEqualTo(-1);
    }

    [Test]
    public async Task ParseMaxParallelism_Zero_ReturnsMinusOne()
    {
        await Assert.That(CompileCudaKernelsTask.ParseMaxParallelism("0")).IsEqualTo(-1);
    }

    [Test]
    public async Task ParseMaxParallelism_Negative_ReturnsMinusOne()
    {
        await Assert.That(CompileCudaKernelsTask.ParseMaxParallelism("-2")).IsEqualTo(-1);
    }

    [Test]
    public async Task ParseMaxParallelism_NonNumeric_ReturnsMinusOne()
    {
        await Assert.That(CompileCudaKernelsTask.ParseMaxParallelism("all")).IsEqualTo(-1);
    }

    // ── IsUpToDate ────────────────────────────────────────────────────────────

    [Test]
    public async Task IsUpToDate_GeneratedMissing_ReturnsFalse()
    {
        string src = Path.GetTempFileName();
        try
        {
            bool result = CompileCudaKernelsTask.IsUpToDate(src, Path.Combine(Path.GetTempPath(), "does_not_exist_xyz.g.cs"));
            await Assert.That(result).IsFalse();
        }
        finally { File.Delete(src); }
    }

    [Test]
    public async Task IsUpToDate_GeneratedNewer_ReturnsTrue()
    {
        string src = Path.GetTempFileName();
        string gen = Path.GetTempFileName();
        try
        {
            File.SetLastWriteTimeUtc(src, DateTime.UtcNow.AddSeconds(-10));
            File.SetLastWriteTimeUtc(gen, DateTime.UtcNow);
            bool result = CompileCudaKernelsTask.IsUpToDate(src, gen);
            await Assert.That(result).IsTrue();
        }
        finally { File.Delete(src); File.Delete(gen); }
    }

    [Test]
    public async Task IsUpToDate_GeneratedOlder_ReturnsFalse()
    {
        string src = Path.GetTempFileName();
        string gen = Path.GetTempFileName();
        try
        {
            File.SetLastWriteTimeUtc(src, DateTime.UtcNow);
            File.SetLastWriteTimeUtc(gen, DateTime.UtcNow.AddSeconds(-10));
            bool result = CompileCudaKernelsTask.IsUpToDate(src, gen);
            await Assert.That(result).IsFalse();
        }
        finally { File.Delete(src); File.Delete(gen); }
    }

    // ── IsStubFile ────────────────────────────────────────────────────────────

    [Test]
    public async Task IsStubFile_FileWithNotImplementedException_ReturnsTrue()
    {
        string f = Path.GetTempFileName();
        try
        {
            File.WriteAllText(f, "// <auto-generated/>\npartial class Foo\n{\n    public partial void Bar()\n        => throw new NotImplementedException(\"Foo.Bar is marked [GpuKernel(NotImplemented=true)]\");\n}");
            await Assert.That(CompileCudaKernelsTask.IsStubFile(f)).IsTrue();
        }
        finally { File.Delete(f); }
    }

    [Test]
    public async Task IsStubFile_FileWithPtx_ReturnsFalse()
    {
        string f = Path.GetTempFileName();
        try
        {
            File.WriteAllText(f, "// <auto-generated/>\nprivate static readonly byte[] _Foo_ptx_encoded = Convert.FromBase64String(\"AAAA\");");
            await Assert.That(CompileCudaKernelsTask.IsStubFile(f)).IsFalse();
        }
        finally { File.Delete(f); }
    }

    [Test]
    public async Task IsStubFile_MissingFile_ReturnsFalse()
    {
        await Assert.That(CompileCudaKernelsTask.IsStubFile(Path.Combine(Path.GetTempPath(), "no_such_file_xyz.g.cs"))).IsFalse();
    }

    // ── EmitBase64Chunks ──────────────────────────────────────────────────────

    [Test]
    public async Task EmitBase64Chunks_ShortString_EmitsSingleChunkWithSemicolon()
    {
        var sb = new System.Text.StringBuilder();
        CompileCudaKernelsTask.EmitBase64Chunks(sb, "AAAA", "  ");
        string result = sb.ToString().Trim();
        await Assert.That(result).IsEqualTo("\"AAAA\");");
    }

    [Test]
    public async Task EmitBase64Chunks_LongString_EmitsMultipleLines()
    {
        var sb = new System.Text.StringBuilder();
        string b64 = new('A', 300);  // longer than the 128-char chunk size
        CompileCudaKernelsTask.EmitBase64Chunks(sb, b64, "");
        string[] lines = sb.ToString().Split('\n', StringSplitOptions.RemoveEmptyEntries);
        await Assert.That(lines.Length).IsGreaterThan(1);
        await Assert.That(lines[^1].Trim()).EndsWith(");");
        await Assert.That(lines[0].Trim()).EndsWith(" +");
    }

    // ── ExtractCudaSignature ──────────────────────────────────────────────────

    [Test]
    public async Task ExtractCudaSignature_SimpleKernel_ReturnsNameAndParamCount()
    {
        const string src = "extern \"C\" __global__ void AddVectors(float* a, float* b, float* c, int n) {}";
        var sig = CompileCudaKernelsTask.ExtractCudaSignature(src);
        await Assert.That(sig).IsNotNull();
        await Assert.That(sig!.Value.Name).IsEqualTo("AddVectors");
        await Assert.That(sig.Value.ParamCount).IsEqualTo(4);
    }

    [Test]
    public async Task ExtractCudaSignature_NoGlobalFunction_ReturnsNull()
    {
        const string src = "void hostHelper(float* x) {}";
        var sig = CompileCudaKernelsTask.ExtractCudaSignature(src);
        await Assert.That(sig).IsNull();
    }

    [Test]
    public async Task ExtractCudaSignature_ZeroParamKernel_ReturnsZeroCount()
    {
        const string src = "__global__ void Ping() {}";
        var sig = CompileCudaKernelsTask.ExtractCudaSignature(src);
        await Assert.That(sig).IsNotNull();
        await Assert.That(sig!.Value.ParamCount).IsEqualTo(0);
    }

    // ── ValidationWarning via ParseKernels ────────────────────────────────────

    [Test]
    public async Task ParseKernels_MatchingNames_NoValidationWarning()
    {
        const string src = """
            using KernelSharp;
            namespace Ns;
            public partial class C {
                [GpuKernel("__global__ void MyKernel(float* a, int n) {}")]
                public partial void MyKernel(CudaBuffer<float> a, int n);
            }
            """;
        var tree = CSharpSyntaxTree.ParseText(src, new CSharpParseOptions(LanguageVersion.Latest));
        var kernels = new List<CompileCudaKernelsTask.KernelInfo>();
        CompileCudaKernelsTask.ParseKernels(tree, string.Empty, kernels);
        await Assert.That(kernels.Count).IsEqualTo(1);
        await Assert.That(kernels[0].ValidationWarning).IsNull();
        await Assert.That(kernels[0].CudaFunctionName).IsEqualTo("MyKernel");
    }

    [Test]
    public async Task ParseKernels_MismatchedName_SetsValidationWarningAndCudaFunctionName()
    {
        const string src = """
            using KernelSharp;
            namespace Ns;
            public partial class C {
                [GpuKernel("__global__ void cuda_add(float* a, int n) {}")]
                public partial void CSharpAdd(CudaBuffer<float> a, int n);
            }
            """;
        var tree = CSharpSyntaxTree.ParseText(src, new CSharpParseOptions(LanguageVersion.Latest));
        var kernels = new List<CompileCudaKernelsTask.KernelInfo>();
        CompileCudaKernelsTask.ParseKernels(tree, string.Empty, kernels);
        await Assert.That(kernels.Count).IsEqualTo(1);
        await Assert.That(kernels[0].ValidationWarning).IsNotNull()
            .Because("a mismatch between CUDA and C# names should produce a warning");
        await Assert.That(kernels[0].CudaFunctionName).IsEqualTo("cuda_add")
            .Because("the actual CUDA function name should be used for cuModuleGetFunction");
    }

    [Test]
    public async Task ParseKernels_ArgCountMismatch_SetsValidationWarning()
    {
        const string src = """
            using KernelSharp;
            namespace Ns;
            public partial class C {
                [GpuKernel("__global__ void Scale(float* a, float s, int n, int extra) {}")]
                public partial void Scale(CudaBuffer<float> a, float s, int n);
            }
            """;
        var tree = CSharpSyntaxTree.ParseText(src, new CSharpParseOptions(LanguageVersion.Latest));
        var kernels = new List<CompileCudaKernelsTask.KernelInfo>();
        CompileCudaKernelsTask.ParseKernels(tree, string.Empty, kernels);
        await Assert.That(kernels.Count).IsEqualTo(1);
        await Assert.That(kernels[0].ValidationWarning).IsNotNull()
            .Because("a CUDA/C# parameter count mismatch should produce a warning");
    }

    [Test]
    public async Task BuildLauncherSource_UsesCudaFunctionName_InEnsureLoaded()
    {
        // Kernel with a CUDA name that differs from the C# method name
        const string src = """
            using KernelSharp;
            namespace Ns;
            public partial class C {
                [GpuKernel("__global__ void cuda_scale(float* a, float s, int n) {}")]
                public partial void Scale(CudaBuffer<float> a, float s, int n);
            }
            """;
        var files = GenerateSources(src);
        await Assert.That(files.Count).IsEqualTo(1);
        await Assert.That(files[0]).Contains("\"cuda_scale\"")
            .Because("cuModuleGetFunction must use the actual CUDA function name, not the C# method name");
    }

    // ── ThreadsPerBlock / BlocksPerGrid ───────────────────────────────────────

    [Test]
    public async Task Generator_DefaultLaunchConfig_Uses256ThreadsAndAutoBlocks()
    {
        // When neither ThreadsPerBlock nor BlocksPerGrid is specified the generator
        // must emit the standard ceil(n/256) block count with 256 threads.
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class K {
                [GpuKernel("extern \"C\" __global__ void DefaultKernel(float* b, int n) {}")]
                public partial void DefaultKernel(CudaBuffer<float> b);
            }}
            """;
        var files = GenerateSources(src);
        string generated = string.Concat(files);

        await Assert.That(generated).Contains("uint _threads = 256;");
        await Assert.That(generated).Contains("uint _blocks = (uint)((_n + (int)_threads - 1) / (int)_threads);");
    }

    [Test]
    public async Task Generator_FixedThreadsPerBlock_EmitsLiteralThreadCount()
    {
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class K {
                [GpuKernel("extern \"C\" __global__ void FixedThreadKernel(float* b, int n) {}", ThreadsPerBlock = 512)]
                public partial void FixedThreadKernel(CudaBuffer<float> b);
            }}
            """;
        var files = GenerateSources(src);
        string generated = string.Concat(files);

        await Assert.That(generated).Contains("uint _threads = 512;");
        await Assert.That(generated).Contains("uint _blocks = (uint)((_n + (int)_threads - 1) / (int)_threads);");
    }

    [Test]
    public async Task Generator_FixedBlocksPerGrid_EmitsLiteralBlockCount()
    {
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class K {
                [GpuKernel("extern \"C\" __global__ void FixedBlockKernel(float* b, int n) {}", BlocksPerGrid = 1)]
                public partial void FixedBlockKernel(CudaBuffer<float> b);
            }}
            """;
        var files = GenerateSources(src);
        string generated = string.Concat(files);

        await Assert.That(generated).Contains("uint _threads = 256;");
        await Assert.That(generated).Contains("uint _blocks = 1;");
    }

    [Test]
    public async Task Generator_FullyFixedLaunchConfig_EmitsNoAutoCompute()
    {
        // When both are set the generator must emit two literal assignments
        // and must NOT emit the auto-compute expression.
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class K {
                [GpuKernel("extern \"C\" __global__ void SingleBlockKernel(float* b, int n) {}",
                    ThreadsPerBlock = 256, BlocksPerGrid = 1)]
                public partial void SingleBlockKernel(CudaBuffer<float> b);
            }}
            """;
        var files = GenerateSources(src);
        string generated = string.Concat(files);

        await Assert.That(generated).Contains("uint _threads = 256;");
        await Assert.That(generated).Contains("uint _blocks = 1;");
        await Assert.That(generated).DoesNotContain("_n + (int)_threads - 1")
            .Because("auto-compute must not appear when both values are fixed");
    }

    [Test]
    public async Task Generator_ThreadsPerBlock_DoesNotAffectOtherKernels()
    {
        // Two kernels in the same class; only one has a custom ThreadsPerBlock.
        const string src = """
            using KernelSharp;
            namespace Ns { public partial class M {
                [GpuKernel("extern \"C\" __global__ void KernelDefault(float* b, int n) {}")]
                public partial void KernelDefault(CudaBuffer<float> b);
                [GpuKernel("extern \"C\" __global__ void KernelCustom(float* b, int n) {}", ThreadsPerBlock = 128)]
                public partial void KernelCustom(CudaBuffer<float> b);
            }}
            """;
        var files = GenerateSources(src);

        string? fileDefault = files.FirstOrDefault(s => s.Contains("KernelDefault") && !s.Contains("KernelCustom"));
        string? fileCustom  = files.FirstOrDefault(s => s.Contains("KernelCustom")  && !s.Contains("KernelDefault"));

        await Assert.That(fileDefault).IsNotNull();
        await Assert.That(fileCustom).IsNotNull();
        await Assert.That(fileDefault!).Contains("uint _threads = 256;");
        await Assert.That(fileCustom!).Contains("uint _threads = 128;");
    }
}

