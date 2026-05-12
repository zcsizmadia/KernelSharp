using KernelSharp;

namespace KernelSharp.Samples;

/// <summary>Samples 01-05 – element-wise vector operations.</summary>
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

    [GpuKernel("""
        extern "C" __global__ void MulVectors(
            const float* __restrict__ a,
            const float* __restrict__ b,
            float*       __restrict__ c,
            int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) c[i] = a[i] * b[i];
        }
        """)]
    public partial void MulVectors(CudaBuffer<float> a, CudaBuffer<float> b, CudaBuffer<float> c);

    [GpuKernel("""
        extern "C" __global__ void SubVectors(
            const float* __restrict__ a,
            const float* __restrict__ b,
            float*       __restrict__ c,
            int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) c[i] = a[i] - b[i];
        }
        """)]
    public partial void SubVectors(CudaBuffer<float> a, CudaBuffer<float> b, CudaBuffer<float> c);

    [GpuKernel("""
        extern "C" __global__ void FmaVectors(
            const float* __restrict__ a,
            const float* __restrict__ b,
            const float* __restrict__ c,
            float*       __restrict__ d,
            int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) d[i] = __fmaf_rn(a[i], b[i], c[i]);
        }
        """)]
    public partial void FmaVectors(CudaBuffer<float> a, CudaBuffer<float> b, CudaBuffer<float> c, CudaBuffer<float> d);

    [GpuKernel("""
        extern "C" __global__ void AbsVector(
            const float* __restrict__ a,
            float*       __restrict__ c,
            int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) c[i] = fabsf(a[i]);
        }
        """)]
    public partial void AbsVector(CudaBuffer<float> a, CudaBuffer<float> c);
}