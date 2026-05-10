// Unit tests for the CudaBuffer, CudaContext, and KernelCache APIs.
// Most require a GPU; pure API tests (dispose, factory, etc.) run anywhere.

using KernelSharp;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;
using TUnit.Core.Exceptions;

namespace KernelSharp.Tests;

public class CudaBufferTests
{
    // ── Factory / dispose (no GPU needed) ─────────────────────────────────────

    [Test]
    public async Task FromPointer_DoesNotOwnMemory_DisposeDoesNotThrow()
    {
        // Use a dummy pointer – we never dereference it, just test lifecycle
        var buf = CudaBuffer<float>.FromPointer(new IntPtr(1234), 10);
        buf.Dispose(); // must not call cuMemFree for non-owned buffers
        // absence of exception = success
        await Task.CompletedTask;
    }

    [Test]
    public Task DoubleDispose_DoesNotThrow()
    {
        var buf = CudaBuffer<float>.FromPointer(new IntPtr(1), 1);
        buf.Dispose();
        buf.Dispose(); // idempotent
        return Task.CompletedTask;
    }

    [Test]
    public async Task DevicePointer_AfterDispose_ThrowsObjectDisposedException()
    {
        var buf = CudaBuffer<float>.FromPointer(new IntPtr(42), 1);
        buf.Dispose();

        await Assert.That(() => { _ = buf.DevicePointer; })
            .Throws<ObjectDisposedException>();
    }

    // ── Alloc / copy (GPU required) ───────────────────────────────────────────

    [Test]
    public async Task AllocateFloat_AndCopyRoundTrip_PreservesValues()
    {
        if (!CudaFixture.HasCuda) throw new SkipTestException("No GPU");
        using var ctx = CudaFixture.RequireCuda();

        float[] src = [1f, 2f, 3f, 4f, 5f];
        float[] dest = new float[5];

        using var buf = CudaBuffer<float>.Allocate(5);
        buf.CopyFromHost(src);
        buf.CopyToHost(dest);

        for (int i = 0; i < 5; i++)
            await Assert.That(dest[i]).IsEqualTo(src[i]).Within(1e-9f);
    }

    [Test]
    public async Task AllocateDouble_AndCopyRoundTrip_PreservesValues()
    {
        if (!CudaFixture.HasCuda) throw new SkipTestException("No GPU");
        using var ctx = CudaFixture.RequireCuda();

        double[] src = [1.1, 2.2, 3.3];
        double[] dest = new double[3];

        using var buf = CudaBuffer<double>.Allocate(3);
        buf.CopyFromHost(src);
        buf.CopyToHost(dest);

        for (int i = 0; i < 3; i++)
            await Assert.That(dest[i]).IsEqualTo(src[i]).Within(1e-15);
    }

    [Test]
    public async Task ByteSize_MatchesExpected()
    {
        if (!CudaFixture.HasCuda) throw new SkipTestException("No GPU");
        using var ctx = CudaFixture.RequireCuda();

        using var buf = CudaBuffer<float>.Allocate(128);
        await Assert.That(buf.ByteSize).IsEqualTo(128L * sizeof(float));
        await Assert.That(buf.Length).IsEqualTo(128);
    }
}

public class CudaContextTests
{
    [Test]
    public async Task DeviceCount_ReturnsNonNegative()
    {
        int count = CudaContext.DeviceCount();
        await Assert.That(count).IsGreaterThanOrEqualTo(0);
    }

    [Test]
    public async Task Initialize_WhenGpuPresent_ReturnsValidHandle()
    {
        if (!CudaFixture.HasCuda) throw new SkipTestException("No GPU");

        using var ctx = CudaContext.Initialize();
        await Assert.That(ctx.Handle).IsNotEqualTo(IntPtr.Zero);
    }

    [Test]
    public async Task Handle_AfterDispose_ThrowsObjectDisposedException()
    {
        if (!CudaFixture.HasCuda) throw new SkipTestException("No GPU");

        var ctx = CudaContext.Initialize();
        ctx.Dispose();

        await Assert.That(() => { _ = ctx.Handle; })
            .Throws<ObjectDisposedException>();
    }
}

public class KernelCacheTests
{
    [Test]
    public async Task GetOrLoadFunction_CalledTwice_ReturnsSameHandle()
    {
        if (!CudaFixture.HasCuda) throw new SkipTestException("No GPU");
        using var ctx = CudaFixture.RequireCuda();

        // Minimal valid kernel source (PTX assembly for SM 7.0+)
        const string kernelSource = @"
.version 7.0
.target sm_70
.address_size 64

.visible .entry NopKernel()
{
    ret;
}
";
        IntPtr h1 = KernelCache.GetOrLoadFunction(kernelSource, "NopKernel");
        IntPtr h2 = KernelCache.GetOrLoadFunction(kernelSource, "NopKernel");

        await Assert.That(h1).IsEqualTo(h2)
            .Because("second call must return the cached handle");
        await Assert.That(h1).IsNotEqualTo(IntPtr.Zero);
    }
}