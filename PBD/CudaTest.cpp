#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
    int deviceCount;

    // 获取可用的 CUDA 设备数目
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        printf("没有找到 CUDA 设备\n");
    }
    else
    {
        printf("找到 %d 个 CUDA 设备\n", deviceCount);
    }

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        printf("\n设备 %d 的属性：\n", i);
        printf("设备名称：%s\n", deviceProp.name);
        printf("设备全局内存大小：%lu MB\n", (unsigned long)deviceProp.totalGlobalMem/1024/1024);
        printf("每个块的共享内存大小：%lu KB\n", (unsigned long)deviceProp.sharedMemPerBlock/1024);
        printf("每个线程块的最大线程数：%d\n", deviceProp.maxThreadsPerBlock);
        printf("设备计算能力：%d.%d\n", deviceProp.major, deviceProp.minor);
        printf("设备支持的 CUDA 版本：%d.%d\n", deviceProp.major, deviceProp.minor);
        printf("设备核心数量：%d\n", deviceProp.multiProcessorCount);
    }

    return 0;
}