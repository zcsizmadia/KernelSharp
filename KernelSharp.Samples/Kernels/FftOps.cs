using KernelSharp;

namespace KernelSharp.Samples;

/// <summary>Samples 21-22 – N-body physics and DCT spectral transform.</summary>
public partial class FftOps
{
    [GpuKernel("""
        #define TILE_SIZE 256
        #define SOFTENING  1e-9f
        #define DT         0.001f

        struct Body { float x, y, z, vx, vy, vz, mass; };

        extern "C" __global__ void NBody(Body* __restrict__ bodies, int n)
        {
            __shared__ Body tile[TILE_SIZE];
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            Body bi = (i < n) ? bodies[i] : Body{};
            float ax = 0, ay = 0, az = 0;

            for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
                int j = t * TILE_SIZE + threadIdx.x;
                tile[threadIdx.x] = (j < n) ? bodies[j] : Body{};
                __syncthreads();
                for (int k = 0; k < TILE_SIZE && t * TILE_SIZE + k < n; ++k) {
                    float dx = tile[k].x - bi.x, dy = tile[k].y - bi.y, dz = tile[k].z - bi.z;
                    float distSq = dx*dx + dy*dy + dz*dz + SOFTENING;
                    float inv3 = rsqrtf(distSq) / distSq * tile[k].mass;
                    ax += dx * inv3; ay += dy * inv3; az += dz * inv3;
                }
                __syncthreads();
            }
            if (i < n) {
                bodies[i].vx += ax * DT; bodies[i].vy += ay * DT; bodies[i].vz += az * DT;
                bodies[i].x  += bodies[i].vx * DT;
                bodies[i].y  += bodies[i].vy * DT;
                bodies[i].z  += bodies[i].vz * DT;
            }
        }
        """)]
    public partial void NBody(CudaBuffer<byte> bodies, int n);

    [GpuKernel("""
        #define PI_F 3.14159265358979323846f

        extern "C" __global__ void DCT(
            const float* __restrict__ x,
            float*       __restrict__ y,
            int n)
        {
            // DCT-II 1D per row — reference for MFCC / compression preprocessing.
            int row = blockIdx.x, k = threadIdx.x;
            if (k >= n) return;
            const float* rx = x + row * n;
            float*       ry = y + row * n;
            float sum = 0.f;
            for (int t = 0; t < n; t++)
                sum += rx[t] * cosf(PI_F * k * (2.f * t + 1.f) / (2.f * n));
            float norm = (k == 0) ? sqrtf(1.f / n) : sqrtf(2.f / n);
            ry[k] = sum * norm;
        }
        """)]
    public partial void DCT(CudaBuffer<float> x, CudaBuffer<float> y, int n);
}