// ClampKernel.cu
// Loaded by KernelSharp via [GpuKernel(SourceFile = "ClampKernel.cu")] — demonstrates
// that kernel source can live in a dedicated .cu file instead of an inline raw string.

extern "C" __global__ void ClampVector(
    const float* __restrict__ x,
    float*       __restrict__ y,
    float lo,
    float hi,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fminf(fmaxf(x[i], lo), hi);
}
