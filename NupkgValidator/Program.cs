using KernelSharp;

namespace NupkgValidator;

public partial class VectorMath
{
    [GpuKernel("""
        extern "C" __global__ void AddVectors(
            const float* __restrict__ a,
            const float* __restrict__ b,
            float*       __restrict__ c,
            int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) c[i] = a[i] + b[i];
        }
        """)]
    public partial void AddVectors(CudaBuffer<float> a, CudaBuffer<float> b, CudaBuffer<float> c);
}

class Program
{
    static void Main(string[] args)
    {
        if (CudaContext.DeviceCount() == 0)
        {
            Console.WriteLine("No CUDA-capable GPU found. Fatbin is compiled and embedded at build time.");
            return;
        }

        using var ctx = CudaContext.Initialize();
        Console.WriteLine("CUDA context initialised.");

        const int N = 1 << 20; // 1 M elements
        float[] ha = new float[N], hb = new float[N];
        for (int i = 0; i < N; i++)
        {
            ha[i] = i;
            hb[i] = N - i;
        }

        using var dA = CudaBuffer<float>.Allocate(N);
        using var dB = CudaBuffer<float>.Allocate(N);
        using var dC = CudaBuffer<float>.Allocate(N);
        dA.CopyFromHost(ha);
        dB.CopyFromHost(hb);

        var vm = new VectorMath();
        vm.AddVectors(dA, dB, dC);
        float[] hc = new float[N];
        dC.CopyToHost(hc);
        Console.WriteLine($"[01] AddVectors  hc[0]={hc[0]:F0}  hc[N-1]={hc[N - 1]:F0}  (expected {N})");
    }
}
