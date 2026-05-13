namespace KernelSharp;

/// <summary>
/// CUDA device attribute identifiers for cuDeviceGetAttribute.
/// Values match the CUdevice_attribute enum in the CUDA Driver API.
/// </summary>
public enum CuDeviceAttribute : int
{
    MaxThreadsPerBlock = 1,
    MaxBlockDimX = 2,
    MaxBlockDimY = 3,
    MaxBlockDimZ = 4,
    MaxGridDimX = 5,
    MaxGridDimY = 6,
    MaxGridDimZ = 7,
    MaxSharedMemoryPerBlock = 8,
    TotalConstantMemory = 9,
    WarpSize = 10,
    MaxRegistersPerBlock = 12,
    ClockRate = 13,
    MultiprocessorCount = 16,
    MemoryClockRate = 36,
    GlobalMemoryBusWidth = 37,
    L2CacheSize = 38,
    MaxThreadsPerMultiProcessor = 39,
    AsyncEngineCount = 40,
    UnifiedAddressing = 41,
    MaxSharedMemoryPerMultiprocessor = 81,
    MaxRegistersPerMultiprocessor = 82,
    GlobalL1CacheSupported = 83,
    LocalL1CacheSupported = 84,
    MultiGpuBoardGroupID = 85,
    ComputeCapabilityMajor = 75,
    ComputeCapabilityMinor = 76,
    ManagedMemory = 88,
    IsMultiGpuBoard = 89,
}