using KernelSharp;

namespace KernelSharp.Samples;

/// <summary>Samples 18-20 – statistics, quantisation, and embedding lookups.</summary>
public partial class StatisticsAndSolvers
{
    [GpuKernel("""
        extern "C" __global__ void TopKMask(
            const float* __restrict__ scores,
            float*       __restrict__ mask,
            int n,
            int k)
        {
            // Partial selection sort in shared memory; masks elements below the k-th value.
            extern __shared__ float topk[];
            int tid = threadIdx.x;
            topk[tid] = (tid < n) ? scores[tid] : -1e38f;
            __syncthreads();

            for (int i = 0; i < k; i++) {
                float mx = -1e38f; int mxi = 0;
                for (int j = 0; j < blockDim.x; j++)
                    if (topk[j] > mx) { mx = topk[j]; mxi = j; }
                if (tid == mxi) { mask[tid] = 1.f; topk[tid] = -1e38f; }
                __syncthreads();
            }
            if (mask[tid] == 0.f) mask[tid] = 0.f;   // zero out non-top-k
        }
        """)]
    public partial void TopKMask(CudaBuffer<float> scores, CudaBuffer<float> mask, int n, int k);

    [GpuKernel("""
        extern "C" __global__ void DequantInt4(
            const unsigned char* __restrict__ packed,
            const float*         __restrict__ scales,
            float*               __restrict__ out,
            int n)
        {
            // Unpack two INT4 nibbles per byte → FP32, scaled per-group.
            // Matches GGUF/GPTQ Q4_0 layout with group_size=32.
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i * 2 + 1 >= n) return;
            unsigned char byte = packed[i];
            int lo = (byte & 0x0F) - 8;
            int hi = (byte >> 4)   - 8;
            float scale = scales[i / 16];   // group of 32 values = 16 bytes
            out[i * 2    ] = lo * scale;
            out[i * 2 + 1] = hi * scale;
        }
        """)]
    public partial void DequantInt4(CudaBuffer<byte> packed, CudaBuffer<float> scales, CudaBuffer<float> output, int n);

    [GpuKernel("""
        extern "C" __global__ void EmbedLookup(
            const int*   __restrict__ tokenIds,
            const float* __restrict__ table,
            float*       __restrict__ out,
            int hiddenDim)
        {
            // Gather embedding rows; one block per token.
            int tok = blockIdx.x;
            int id  = tokenIds[tok];
            const float* row = table + id * hiddenDim;
            float*       dst = out   + tok * hiddenDim;
            for (int d = threadIdx.x; d < hiddenDim; d += blockDim.x)
                dst[d] = row[d];
        }
        """)]
    public partial void EmbedLookup(CudaBuffer<int> tokenIds, CudaBuffer<float> table, CudaBuffer<float> output, int hiddenDim);
}