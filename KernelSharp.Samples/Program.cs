using KernelSharp;
using KernelSharp.Samples;

Console.WriteLine("KernelSharp Samples - GPU Kernel Demo");
Console.WriteLine("=======================================");

if (CudaContext.DeviceCount() == 0)
{
    Console.WriteLine("No CUDA-capable GPU found. Fatbin is compiled and embedded at build time.");
    return;
}

using var ctx = CudaContext.Initialize();
Console.WriteLine("CUDA context initialised.");

const int N = 1 << 20; // 1 M elements
float[] ha = new float[N], hb = new float[N];
for (int i = 0; i < N; i++) { ha[i] = i; hb[i] = N - i; }

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

vm.MulVectors(dA, dB, dC);
dC.CopyToHost(hc);
Console.WriteLine($"[02] MulVectors  hc[0]={hc[0]:F0}  (expected 0)");

// -- Activations
float[] hx = new float[N];
for (int i = 0; i < N; i++)
{
    hx[i] = (float)i / N - 0.5f;
}

using var dX = CudaBuffer<float>.Allocate(N);
using var dY = CudaBuffer<float>.Allocate(N);
dX.CopyFromHost(hx);

var act = new ActivationFunctions();
act.ReLU(dX, dY);
float[] hy = new float[N];
dY.CopyToHost(hy);
Console.WriteLine($"[06] ReLU  hy[0]={hy[0]:F3} (expected 0)  hy[N-1]={hy[N - 1]:F4}");

act.GELU(dX, dY);
dY.CopyToHost(hy);
Console.WriteLine($"[07] GELU  hy[N/2]={hy[N / 2]:F4}");

act.SiLU(dX, dY);
dY.CopyToHost(hy);
Console.WriteLine($"[08] SiLU  hy[N/2]={hy[N / 2]:F4}");

// -- Reduction
float[] hOnes = new float[N];
Array.Fill(hOnes, 1f);
using var dOnes = CudaBuffer<float>.Allocate(N);
using var dRes = CudaBuffer<float>.Allocate(1);
dOnes.CopyFromHost(hOnes);
var red = new ReductionOps();
red.WarpReduce(dOnes, dRes);
float[] hResult = new float[1];
dRes.CopyToHost(hResult);
Console.WriteLine($"[15] WarpReduce  result={hResult[0]:F0} (expected {N})");

Console.WriteLine("\nAll samples completed successfully.");