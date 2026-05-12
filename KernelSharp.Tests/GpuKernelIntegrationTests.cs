// GPU integration tests – run only when a CUDA device is present.
// Uses the sample kernel classes compiled by the source generator.

using KernelSharp;
using KernelSharp.Samples;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;
using TUnit.Core.Exceptions;

namespace KernelSharp.Tests;

public class GpuKernelIntegrationTests
{
    private const int N = 1 << 16; // 64 K – fast even on low-end GPUs
    private const float Eps = 1e-4f;

    // ── Shared context (created once per class) ───────────────────────────────

    private static CudaContext? _ctx;

    [Before(Class)]
    public static void SetupCuda()
    {
        if (!CudaFixture.HasCuda)
        {
            return;
        }

        _ctx = CudaContext.Initialize();
    }

    [After(Class)]
    public static void TeardownCuda()
    {
        _ctx?.Dispose();
        _ctx = null;
    }

    // Make the shared context current on whichever thread TUnit picks for this test.
    // CUDA Driver API contexts are thread-local; without this, parallel tests that run
    // on different pool threads will get CUDA_ERROR_INVALID_CONTEXT.
    [Before(Test)]
    public void EnsureContextCurrent()
    {
        if (CudaFixture.HasCuda)
        {
            _ctx?.MakeCurrent();
        }
    }

    // ── Vector Math ───────────────────────────────────────────────────────────

    [Test]
    public async Task AddVectors_ProducesCorrectResult()
    {
        if (!CudaFixture.HasCuda)
        {
            throw new SkipTestException("No GPU");
        }

        float[] ha = new float[N], hb = new float[N], hc = new float[N];
        for (int i = 0; i < N; i++) { ha[i] = i; hb[i] = N - i; }

        using var dA = CudaBuffer<float>.Allocate(N);
        using var dB = CudaBuffer<float>.Allocate(N);
        using var dC = CudaBuffer<float>.Allocate(N);

        dA.CopyFromHost(ha);
        dB.CopyFromHost(hb);

        new VectorMath().AddVectors(dA, dB, dC);
        dC.CopyToHost(hc);

        // Every element should equal N
        for (int i = 0; i < N; i++)
        {
            await Assert.That(hc[i]).IsEqualTo(N).Within(Eps);
        }
    }

    [Test]
    public async Task SubVectors_ProducesCorrectResult()
    {
        if (!CudaFixture.HasCuda)
        {
            throw new SkipTestException("No GPU");
        }

        float[] ha = new float[N], hb = new float[N], hc = new float[N];
        for (int i = 0; i < N; i++) { ha[i] = 2f * i; hb[i] = i; }

        using var dA = CudaBuffer<float>.Allocate(N);
        using var dB = CudaBuffer<float>.Allocate(N);
        using var dC = CudaBuffer<float>.Allocate(N);

        dA.CopyFromHost(ha); dB.CopyFromHost(hb);
        new VectorMath().SubVectors(dA, dB, dC);
        dC.CopyToHost(hc);

        for (int i = 0; i < N; i++)
        {
            await Assert.That(hc[i]).IsEqualTo(i).Within(Eps);
        }
    }

    [Test]
    public async Task MulVectors_ProducesCorrectResult()
    {
        if (!CudaFixture.HasCuda)
        {
            throw new SkipTestException("No GPU");
        }

        float[] ha = new float[N], hb = new float[N], hc = new float[N];
        for (int i = 0; i < N; i++) { ha[i] = i + 1f; hb[i] = 2f; }

        using var dA = CudaBuffer<float>.Allocate(N);
        using var dB = CudaBuffer<float>.Allocate(N);
        using var dC = CudaBuffer<float>.Allocate(N);

        dA.CopyFromHost(ha); dB.CopyFromHost(hb);
        new VectorMath().MulVectors(dA, dB, dC);
        dC.CopyToHost(hc);

        for (int i = 0; i < N; i++)
        {
            await Assert.That(hc[i]).IsEqualTo(2f * (i + 1f)).Within(Eps);
        }
    }

    [Test]
    public async Task AbsVector_NegativeInputs_AllPositive()
    {
        if (!CudaFixture.HasCuda)
        {
            throw new SkipTestException("No GPU");
        }

        float[] hx = [.. Enumerable.Range(0, N).Select(i => (float)(i - N / 2))];
        float[] hy = new float[N];

        using var dX = CudaBuffer<float>.Allocate(N);
        using var dY = CudaBuffer<float>.Allocate(N);

        dX.CopyFromHost(hx);
        new VectorMath().AbsVector(dX, dY);
        dY.CopyToHost(hy);

        for (int i = 0; i < N; i++)
        {
            await Assert.That(hy[i]).IsGreaterThanOrEqualTo(0f);
        }
    }

    // ── Activation Functions ──────────────────────────────────────────────────

    [Test]
    public async Task ReLU_NegativeInputsAreZeroed()
    {
        if (!CudaFixture.HasCuda)
        {
            throw new SkipTestException("No GPU");
        }

        float[] hx = [.. Enumerable.Range(0, N).Select(i => (float)(i - N / 2))];
        float[] hy = new float[N];

        using var dX = CudaBuffer<float>.Allocate(N);
        using var dY = CudaBuffer<float>.Allocate(N);

        dX.CopyFromHost(hx);
        new ActivationFunctions().ReLU(dX, dY);
        dY.CopyToHost(hy);

        for (int i = 0; i < N; i++)
        {
            await Assert.That(hy[i]).IsGreaterThanOrEqualTo(0f);
        }
    }

    [Test]
    public async Task Sigmoid_OutputBoundedBetweenZeroAndOne()
    {
        if (!CudaFixture.HasCuda)
        {
            throw new SkipTestException("No GPU");
        }

        float[] hx = [.. Enumerable.Range(0, N).Select(i => (float)(i - N / 2) * 0.001f)];
        float[] hy = new float[N];

        using var dX = CudaBuffer<float>.Allocate(N);
        using var dY = CudaBuffer<float>.Allocate(N);

        dX.CopyFromHost(hx);
        new ActivationFunctions().Sigmoid(dX, dY);
        dY.CopyToHost(hy);

        for (int i = 0; i < N; i++)
        {
            await Assert.That(hy[i]).IsGreaterThan(-Eps);
            await Assert.That(hy[i]).IsLessThan(1f + Eps);
        }
    }

    [Test]
    public async Task TanhActivation_OutputBoundedBetweenMinusOneAndOne()
    {
        if (!CudaFixture.HasCuda)
        {
            throw new SkipTestException("No GPU");
        }

        float[] hx = [.. Enumerable.Range(0, N).Select(i => (float)(i - N / 2) * 0.001f)];
        float[] hy = new float[N];

        using var dX = CudaBuffer<float>.Allocate(N);
        using var dY = CudaBuffer<float>.Allocate(N);

        dX.CopyFromHost(hx);
        new ActivationFunctions().TanhActivation(dX, dY);
        dY.CopyToHost(hy);

        for (int i = 0; i < N; i++)
        {
            await Assert.That(hy[i]).IsGreaterThan(-1f - Eps);
            await Assert.That(hy[i]).IsLessThan(1f + Eps);
        }
    }

    // ── Reductions ────────────────────────────────────────────────────────────

    [Test]
    public async Task SumReduce_AllOnes_EqualsN()
    {
        if (!CudaFixture.HasCuda)
        {
            throw new SkipTestException("No GPU");
        }

        float[] hones = [.. Enumerable.Repeat(1f, N)];
        float[] hout = new float[1];

        using var dIn = CudaBuffer<float>.Allocate(N);
        using var dOut = CudaBuffer<float>.Allocate(1);

        dIn.CopyFromHost(hones);
        new ReductionOps().SumReduce(dIn, dOut);
        dOut.CopyToHost(hout);

        await Assert.That(hout[0]).IsEqualTo(N).Within(N * Eps);
    }

    [Test]
    public async Task MaxReduce_ReturnsMaxElement()
    {
        if (!CudaFixture.HasCuda)
        {
            throw new SkipTestException("No GPU");
        }

        float[] hx = [.. Enumerable.Range(0, N).Select(i => (float)i)];
        float[] hout = new float[1];

        using var dIn = CudaBuffer<float>.Allocate(N);
        using var dOut = CudaBuffer<float>.Allocate(1);

        dIn.CopyFromHost(hx);
        new ReductionOps().MaxReduce(dIn, dOut);
        dOut.CopyToHost(hout);

        await Assert.That(hout[0]).IsEqualTo(N - 1f).Within(Eps);
    }

    [Test]
    public async Task MinReduce_ReturnsMinElement()
    {
        if (!CudaFixture.HasCuda)
        {
            throw new SkipTestException("No GPU");
        }

        float[] hx = [.. Enumerable.Range(0, N).Select(i => (float)i + 1f)];
        float[] hout = new float[1];

        using var dIn = CudaBuffer<float>.Allocate(N);
        using var dOut = CudaBuffer<float>.Allocate(1);

        // MinReduce uses CAS-based float min; output must start at +infinity.
        dOut.CopyFromHost([float.MaxValue]);
        dIn.CopyFromHost(hx);
        new ReductionOps().MinReduce(dIn, dOut);
        dOut.CopyToHost(hout);

        await Assert.That(hout[0]).IsEqualTo(1f).Within(Eps);
    }

    [Test]
    public async Task DotProduct_UnitVectors_EqualsN()
    {
        if (!CudaFixture.HasCuda)
        {
            throw new SkipTestException("No GPU");
        }

        float[] hones = [.. Enumerable.Repeat(1f, N)];
        float[] hout = new float[1];

        using var dA = CudaBuffer<float>.Allocate(N);
        using var dB = CudaBuffer<float>.Allocate(N);
        using var dOut = CudaBuffer<float>.Allocate(1);

        dA.CopyFromHost(hones); dB.CopyFromHost(hones);
        new ReductionOps().DotProduct(dA, dB, dOut);
        dOut.CopyToHost(hout);

        await Assert.That(hout[0]).IsEqualTo(N).Within(N * Eps);
    }

    [Test]
    public async Task L2Norm_UnitVector_EqualsSquareRootOfN()
    {
        if (!CudaFixture.HasCuda)
        {
            throw new SkipTestException("No GPU");
        }

        float[] hones = [.. Enumerable.Repeat(1f, N)];
        float[] hout = new float[1];

        using var dX = CudaBuffer<float>.Allocate(N);
        using var dOut = CudaBuffer<float>.Allocate(1);

        dX.CopyFromHost(hones);
        new ReductionOps().L2Norm(dX, dOut);
        dOut.CopyToHost(hout);

        // L2Norm kernel accumulates sum-of-squares via atomicAdd; take sqrt on the host.
        float norm = (float)Math.Sqrt(hout[0]);
        await Assert.That(norm).IsEqualTo((float)Math.Sqrt(N)).Within(Eps * (float)Math.Sqrt(N));
    }

    // ── Linear Algebra ────────────────────────────────────────────────────────

    [Test]
    public async Task Gemm_IdentityMatrix_ReturnsOriginalMatrix()
    {
        if (!CudaFixture.HasCuda)
        {
            throw new SkipTestException("No GPU");
        }

        const int M = 64;
        float[] hA = new float[M * M];
        float[] hI = new float[M * M]; // identity
        float[] hC = new float[M * M];

        for (int r = 0; r < M; r++)
        {
            hA[r * M + r] = 1f;  // A = identity too for simplicity
            hI[r * M + r] = 1f;
        }

        using var dA = CudaBuffer<float>.Allocate(M * M);
        using var dB = CudaBuffer<float>.Allocate(M * M);
        using var dC = CudaBuffer<float>.Allocate(M * M);

        dA.CopyFromHost(hA); dB.CopyFromHost(hI);
        new LinAlgOps().Gemm(dA, dB, dC);
        dC.CopyToHost(hC);

        // I * I = I
        for (int r = 0; r < M; r++)
        {
            for (int c = 0; c < M; c++)
            {
                float expected = r == c ? 1f : 0f;
                await Assert.That(hC[r * M + c]).IsEqualTo(expected).Within(Eps);
            }
        }
    }

    [Test]
    public async Task Transpose_4x4_Matrix_ProducesCorrectResult()
    {
        if (!CudaFixture.HasCuda)
        {
            throw new SkipTestException("No GPU");
        }

        // Transpose kernel requires a square M×M matrix (n = M*M).
        // A = 4×4 identity; Aᵀ = identity.
        const int M = 4;
        float[] hA = new float[M * M];
        float[] hAt = new float[M * M];
        for (int r = 0; r < M; r++)
        {
            hA[r * M + r] = 1f;  // identity
        }

        using var dA = CudaBuffer<float>.Allocate(M * M);
        using var dAt = CudaBuffer<float>.Allocate(M * M);

        dA.CopyFromHost(hA);
        new LinAlgOps().Transpose(dA, dAt);
        dAt.CopyToHost(hAt);

        // Aᵀ of identity is identity
        for (int r = 0; r < M; r++)
        {
            for (int c = 0; c < M; c++)
            {
                await Assert.That(hAt[r * M + c]).IsEqualTo(hA[r * M + c]).Within(Eps);
            }
        }
    }

    [Test]
    public async Task PrefixScan_AllOnes_ProducesNaturalNumbers()
    {
        if (!CudaFixture.HasCuda)
        {
            throw new SkipTestException("No GPU");
        }

        const int Len = 256; // single-block (≤256 elements fit in one 256-thread block)
        float[] hx = [.. Enumerable.Repeat(1f, Len)];
        float[] hy = new float[Len];

        using var dX = CudaBuffer<float>.Allocate(Len);
        using var dY = CudaBuffer<float>.Allocate(Len);

        dX.CopyFromHost(hx);
        new LinAlgOps().PrefixScan(dX, dY);
        dY.CopyToHost(hy);

        for (int i = 0; i < Len; i++)
        {
            await Assert.That(hy[i]).IsEqualTo(i + 1f).Within(Eps);
        }
    }
}