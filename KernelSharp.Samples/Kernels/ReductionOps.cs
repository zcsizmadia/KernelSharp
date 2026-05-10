using KernelSharp;

namespace KernelSharp.Samples;

/// <summary>Samples 12-14 – GPU reduction primitives.</summary>
public partial class ReductionOps
{
    [GpuKernel("""
        extern "C" __global__ void WarpReduce(
            const float* __restrict__ input,
            float*       __restrict__ output,
            int n)
        {
            // Each warp reduces 32 elements with shuffle, then one thread atomicAdds.
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            float v = (i < n) ? input[i] : 0.f;
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down_sync(0xffffffff, v, offset);
            if ((threadIdx.x & 31) == 0)
                atomicAdd(output, v);
        }
        """, NotImplemented = true)]
    public partial void WarpReduce(CudaBuffer<float> input, CudaBuffer<float> output);

    [GpuKernel("""
        extern "C" __global__ void WarpMax(
            const float* __restrict__ input,
            float*       __restrict__ output,
            int n)
        {
            // Per-warp maximum via shuffle, then global atomic max (via CAS).
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            float v = (i < n) ? input[i] : -1e38f;
            for (int offset = 16; offset > 0; offset >>= 1)
                v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
            if ((threadIdx.x & 31) == 0) {
                unsigned int* addr = (unsigned int*)output;
                unsigned int old = *addr, assumed;
                do {
                    assumed = old;
                    float cur = __uint_as_float(assumed);
                    if (v <= cur) break;
                    old = atomicCAS(addr, assumed, __float_as_uint(v));
                } while (assumed != old);
            }
        }
        """, NotImplemented = true)]
    public partial void WarpMax(CudaBuffer<float> input, CudaBuffer<float> output);

    [GpuKernel("""
        extern "C" __global__ void PrefixSum(
            const float* __restrict__ input,
            float*       __restrict__ output,
            int n)
        {
            // Blelloch exclusive scan — O(n) work, O(log n) steps.
            extern __shared__ float smem[];
            int tid = threadIdx.x;
            smem[tid] = (tid < n) ? input[tid] : 0.f;
            __syncthreads();

            // Up-sweep (reduce)
            for (int d = 1; d < blockDim.x; d <<= 1) {
                int idx = (tid + 1) * (d << 1) - 1;
                if (idx < blockDim.x) smem[idx] += smem[idx - d];
                __syncthreads();
            }
            if (tid == 0) smem[blockDim.x - 1] = 0.f;
            __syncthreads();

            // Down-sweep
            for (int d = blockDim.x >> 1; d >= 1; d >>= 1) {
                int idx = (tid + 1) * (d << 1) - 1;
                if (idx < blockDim.x) {
                    float t = smem[idx - d];
                    smem[idx - d] = smem[idx];
                    smem[idx]    += t;
                }
                __syncthreads();
            }
            if (tid < n) output[tid] = smem[tid];
        }
        """, NotImplemented = true)]
    public partial void PrefixSum(CudaBuffer<float> input, CudaBuffer<float> output);

    [GpuKernel("""
        extern "C" __global__ void SumReduce(
            const float* __restrict__ input,
            float*       __restrict__ output,
            int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            float v = (i < n) ? input[i] : 0.f;
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down_sync(0xffffffff, v, offset);
            if ((threadIdx.x & 31) == 0)
                atomicAdd(output, v);
        }
        """, NotImplemented = true)]
    public partial void SumReduce(CudaBuffer<float> input, CudaBuffer<float> output);

    [GpuKernel("""
        extern "C" __global__ void MaxReduce(
            const float* __restrict__ input,
            float*       __restrict__ output,
            int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            float v = (i < n) ? input[i] : -1e38f;
            for (int offset = 16; offset > 0; offset >>= 1)
                v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
            if ((threadIdx.x & 31) == 0) {
                unsigned int* addr = (unsigned int*)output;
                unsigned int old = *addr, assumed;
                do {
                    assumed = old;
                    float cur = __uint_as_float(assumed);
                    if (v <= cur) break;
                    old = atomicCAS(addr, assumed, __float_as_uint(v));
                } while (assumed != old);
            }
        }
        """, NotImplemented = true)]
    public partial void MaxReduce(CudaBuffer<float> input, CudaBuffer<float> output);

    [GpuKernel("""
        extern "C" __global__ void MinReduce(
            const float* __restrict__ input,
            float*       __restrict__ output,
            int n)
        {
            // output must be pre-initialised to +FLT_MAX by the caller.
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            float v = (i < n) ? input[i] : 3.402823466e+38f;
            for (int offset = 16; offset > 0; offset >>= 1)
                v = fminf(v, __shfl_down_sync(0xffffffff, v, offset));
            if ((threadIdx.x & 31) == 0) {
                unsigned int* addr = (unsigned int*)output;
                unsigned int old = *addr, assumed;
                do {
                    assumed = old;
                    float cur = __uint_as_float(assumed);
                    if (v >= cur) break;
                    old = atomicCAS(addr, assumed, __float_as_uint(v));
                } while (assumed != old);
            }
        }
        """, NotImplemented = true)]
    public partial void MinReduce(CudaBuffer<float> input, CudaBuffer<float> output);

    [GpuKernel("""
        extern "C" __global__ void DotProduct(
            const float* __restrict__ a,
            const float* __restrict__ b,
            float*       __restrict__ output,
            int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            float v = (i < n) ? a[i] * b[i] : 0.f;
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down_sync(0xffffffff, v, offset);
            if ((threadIdx.x & 31) == 0)
                atomicAdd(output, v);
        }
        """, NotImplemented = true)]
    public partial void DotProduct(CudaBuffer<float> a, CudaBuffer<float> b, CudaBuffer<float> output);

    [GpuKernel("""
        extern "C" __global__ void L2Norm(
            const float* __restrict__ x,
            float*       __restrict__ output,
            int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            float v = (i < n) ? x[i] * x[i] : 0.f;
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down_sync(0xffffffff, v, offset);
            if ((threadIdx.x & 31) == 0)
                atomicAdd(output, v);
            // The host must take sqrt of the result after the kernel finishes.
        }
        """, NotImplemented = true)]
    public partial void L2Norm(CudaBuffer<float> x, CudaBuffer<float> output);
}