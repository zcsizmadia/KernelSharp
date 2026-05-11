using KernelSharp;

namespace KernelSharp.Samples;

/// <summary>Samples 15-17 – attention and positional embedding kernels.</summary>
public partial class LinAlgOps
{
    [GpuKernel("""
        extern "C" __global__ void RoPE(
            float* __restrict__ q,
            float* __restrict__ k,
            int seqLen,
            int headDim)
        {
            // Rotary Position Embedding (LLaMA, Falcon).
            // Each thread handles one (pos, dim/2) pair.
            int pos  = blockIdx.x;
            int pair = threadIdx.x;   // pair < headDim/2
            if (pair >= headDim / 2) return;

            float theta = 1.f / powf(10000.f, 2.f * pair / (float)headDim);
            float angle = (float)pos * theta;
            float cosA  = cosf(angle), sinA = sinf(angle);

            int base = pos * headDim + 2 * pair;
            float q0 = q[base], q1 = q[base + 1];
            q[base]     = q0 * cosA - q1 * sinA;
            q[base + 1] = q0 * sinA + q1 * cosA;

            float k0 = k[base], k1 = k[base + 1];
            k[base]     = k0 * cosA - k1 * sinA;
            k[base + 1] = k0 * sinA + k1 * cosA;
        }
        """)]
    public partial void RoPE(CudaBuffer<float> q, CudaBuffer<float> k, int seqLen, int headDim);

    [GpuKernel("""
        extern "C" __global__ void AttnScores(
            const float* __restrict__ q,
            const float* __restrict__ k,
            float*       __restrict__ scores,
            int seqLen,
            int headDim)
        {
            // Scaled dot-product with causal mask.
            // Grid: (seqLen, seqLen); each thread: one (query_pos, key_pos) score.
            int qi = blockIdx.x, ki = blockIdx.y;
            float scale = rsqrtf((float)headDim);
            float s = 0.f;
            for (int d = 0; d < headDim; d++)
                s += q[qi * headDim + d] * k[ki * headDim + d];
            // Causal mask
            scores[qi * seqLen + ki] = (ki <= qi) ? s * scale : -1e38f;
        }
        """)]
    public partial void AttnScores(CudaBuffer<float> q, CudaBuffer<float> k, CudaBuffer<float> scores, int seqLen, int headDim);

    [GpuKernel("""
        extern "C" __global__ void AttnOutput(
            const float* __restrict__ attnWeights,
            const float* __restrict__ v,
            float*       __restrict__ out,
            int seqLen,
            int headDim)
        {
            // Weighted aggregation of value vectors.
            // Grid: (seqLen,); one block per query position.
            int qi = blockIdx.x, d = threadIdx.x;
            if (d >= headDim) return;
            float acc = 0.f;
            for (int ki = 0; ki < seqLen; ki++)
                acc += attnWeights[qi * seqLen + ki] * v[ki * headDim + d];
            out[qi * headDim + d] = acc;
        }
        """)]
    public partial void AttnOutput(CudaBuffer<float> attnWeights, CudaBuffer<float> v, CudaBuffer<float> output, int seqLen, int headDim);

    [GpuKernel("""
        extern "C" __global__ void Gemm(
            const float* __restrict__ a,
            const float* __restrict__ b,
            float*       __restrict__ c,
            int n)
        {
            // Square matrix multiply: n = M*M, M = sqrt(n).
            // Each thread computes one output element c[row, col] = dot(A[row,:], B[:,col]).
            int m   = (int)sqrtf((float)n);
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;
            int row = idx / m, col = idx % m;
            float acc = 0.f;
            for (int k = 0; k < m; k++)
                acc += a[row * m + k] * b[k * m + col];
            c[idx] = acc;
        }
        """)]
    public partial void Gemm(CudaBuffer<float> a, CudaBuffer<float> b, CudaBuffer<float> c);

    [GpuKernel("""
        extern "C" __global__ void Transpose(
            const float* __restrict__ a,
            float*       __restrict__ at,
            int n)
        {
            // Square matrix transpose: n = M*M, M = sqrt(n).
            int m   = (int)sqrtf((float)n);
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;
            int row = idx / m, col = idx % m;
            at[col * m + row] = a[row * m + col];
        }
        """)]
    public partial void Transpose(CudaBuffer<float> a, CudaBuffer<float> at);

    [GpuKernel("""
        extern "C" __global__ void PrefixScan(
            const float* __restrict__ x,
            float*       __restrict__ y,
            int n)
        {
            // Inclusive prefix sum – single-block, shared memory Hillis-Steele.
            extern __shared__ float smem[];
            int tid = threadIdx.x;
            smem[tid] = (tid < n) ? x[tid] : 0.f;
            __syncthreads();
            for (int d = 1; d < blockDim.x; d <<= 1) {
                float v = (tid >= d) ? smem[tid - d] : 0.f;
                __syncthreads();
                smem[tid] += v;
                __syncthreads();
            }
            if (tid < n) y[tid] = smem[tid];
        }
        """)]
    public partial void PrefixScan(CudaBuffer<float> x, CudaBuffer<float> y);
}