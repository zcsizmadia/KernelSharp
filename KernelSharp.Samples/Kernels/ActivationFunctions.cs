using KernelSharp;

namespace KernelSharp.Samples;

/// <summary>Samples 06-11 – neural-network activation functions.</summary>
public partial class ActivationFunctions
{
    [GpuKernel("""
        extern "C" __global__ void ReLU(
            const float* __restrict__ x,
            float*       __restrict__ y,
            int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) y[i] = x[i] > 0.f ? x[i] : 0.f;
        }
        """)]
    public partial void ReLU(CudaBuffer<float> x, CudaBuffer<float> y);

    [GpuKernel("""
        extern "C" __global__ void GELU(
            const float* __restrict__ x,
            float*       __restrict__ y,
            int n)
        {
            // GPT-2/3 tanh approximation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
            const float kA = 0.7978845608f; // sqrt(2/pi)
            const float kB = 0.044715f;
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                float v = x[i];
                float inner = kA * (v + kB * v * v * v);
                y[i] = 0.5f * v * (1.f + tanhf(inner));
            }
        }
        """)]
    public partial void GELU(CudaBuffer<float> x, CudaBuffer<float> y);

    [GpuKernel("""
        extern "C" __global__ void SiLU(
            const float* __restrict__ x,
            float*       __restrict__ y,
            int n)
        {
            // SiLU / Swish: x * sigmoid(x)  — used in LLaMA gate projections
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) { float v = x[i]; y[i] = v / (1.f + expf(-v)); }
        }
        """)]
    public partial void SiLU(CudaBuffer<float> x, CudaBuffer<float> y);

    [GpuKernel("""
        extern "C" __global__ void SwiGLU(
            const float* __restrict__ gate,
            const float* __restrict__ up,
            float*       __restrict__ y,
            int n)
        {
            // SwiGLU = SiLU(gate) * up  — fused MLP gate used in LLaMA 2/3, Mistral
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                float g = gate[i];
                y[i] = (g / (1.f + expf(-g))) * up[i];
            }
        }
        """)]
    public partial void SwiGLU(CudaBuffer<float> gate, CudaBuffer<float> up, CudaBuffer<float> y);

    [GpuKernel("""
        extern "C" __global__ void RMSNorm(
            const float* __restrict__ x,
            const float* __restrict__ weight,
            float*       __restrict__ y,
            int n,
            float eps)
        {
            // One block per row; shared reduction to compute RMS.
            extern __shared__ float smem[];
            int row = blockIdx.x;
            const float* row_x = x + row * n;
            float*       row_y = y + row * n;

            float ss = 0.f;
            for (int j = threadIdx.x; j < n; j += blockDim.x)
                ss += row_x[j] * row_x[j];
            smem[threadIdx.x] = ss;
            __syncthreads();
            for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
                if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
                __syncthreads();
            }
            float rms = rsqrtf(smem[0] / (float)n + eps);
            for (int j = threadIdx.x; j < n; j += blockDim.x)
                row_y[j] = row_x[j] * rms * weight[j];
        }
        """)]
    public partial void RMSNorm(CudaBuffer<float> x, CudaBuffer<float> weight, CudaBuffer<float> y, int n, float eps);

    [GpuKernel("""
        extern "C" __global__ void Softmax(
            const float* __restrict__ x,
            float*       __restrict__ y,
            int n)
        {
            // Numerically stable: find max, subtract, exponentiate, normalise.
            extern __shared__ float smem[];
            int row = blockIdx.x;
            const float* rx = x + row * n;
            float*       ry = y + row * n;

            float mx = -1e38f;
            for (int j = threadIdx.x; j < n; j += blockDim.x)
                mx = fmaxf(mx, rx[j]);
            smem[threadIdx.x] = mx;
            __syncthreads();
            for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
                if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x+s]);
                __syncthreads();
            }
            mx = smem[0];

            float sum = 0.f;
            for (int j = threadIdx.x; j < n; j += blockDim.x) {
                ry[j] = expf(rx[j] - mx);
                sum  += ry[j];
            }
            smem[threadIdx.x] = sum;
            __syncthreads();
            for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
                if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
                __syncthreads();
            }
            sum = smem[0];
            for (int j = threadIdx.x; j < n; j += blockDim.x)
                ry[j] /= sum;
        }
        """)]
    public partial void Softmax(CudaBuffer<float> x, CudaBuffer<float> y, int n);

    [GpuKernel("""
        extern "C" __global__ void Sigmoid(
            const float* __restrict__ x,
            float*       __restrict__ y,
            int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) y[i] = 1.f / (1.f + expf(-x[i]));
        }
        """)]
    public partial void Sigmoid(CudaBuffer<float> x, CudaBuffer<float> y);

    [GpuKernel("""
        extern "C" __global__ void TanhActivation(
            const float* __restrict__ x,
            float*       __restrict__ y,
            int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) y[i] = tanhf(x[i]);
        }
        """, NotImplemented = true)]
    public partial void TanhActivation(CudaBuffer<float> x, CudaBuffer<float> y);
}