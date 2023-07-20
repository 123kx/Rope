#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
    int deviceCount;

    // ��ȡ���õ� CUDA �豸��Ŀ
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        printf("û���ҵ� CUDA �豸\n");
    }
    else
    {
        printf("�ҵ� %d �� CUDA �豸\n", deviceCount);
    }

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        printf("\n�豸 %d �����ԣ�\n", i);
        printf("�豸���ƣ�%s\n", deviceProp.name);
        printf("�豸ȫ���ڴ��С��%lu MB\n", (unsigned long)deviceProp.totalGlobalMem/1024/1024);
        printf("ÿ����Ĺ����ڴ��С��%lu KB\n", (unsigned long)deviceProp.sharedMemPerBlock/1024);
        printf("ÿ���߳̿������߳�����%d\n", deviceProp.maxThreadsPerBlock);
        printf("�豸����������%d.%d\n", deviceProp.major, deviceProp.minor);
        printf("�豸֧�ֵ� CUDA �汾��%d.%d\n", deviceProp.major, deviceProp.minor);
        printf("�豸����������%d\n", deviceProp.multiProcessorCount);
    }

    return 0;
}