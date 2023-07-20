#pragma once

#include <tuple>

#include <glm/glm.hpp>

#include <cuda_runtime.h> 
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "cuda/helper_cuda.h"
#include "cuda/helper_math.h"

#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/sort.h>

#define CONST(type)				const type const
// grid划分成1维，block划分为1维
//计算ID：通过将blockIdx.x（块在x方向上的索引）乘以blockDim.x（一个块中的线程数），
//再加上threadIdx.x（线程在块中的索引），计算出每个线程的唯一标识符id。
//二维线程中的列号
//Dim 数从1开始标，线程数Idx从0开始标。
#define GET_CUDA_ID(id, maxID) 	uint id = blockIdx.x * blockDim.x + threadIdx.x; if (id >= maxID) return
#define GET_CUDA_ID_NO_RETURN(id, maxID) 	uint id = blockIdx.x * blockDim.x + threadIdx.x
#define EPSILON					1e-6f

#ifdef __CUDACC__ 
#define CUDA_CALL(func, totalThreads)  \
	if (totalThreads == 0) return; \
	uint func ## _numBlocks, func ## _numThreads; \
	ComputeGridSize(totalThreads, func ## _numBlocks, func ## _numThreads); \
	func <<<func ## _numBlocks, func ## _numThreads >>>
#define CUDA_CALL_S(func, totalThreads, stream)  \
	if (totalThreads == 0) return; \
	uint func ## _numBlocks, func ## _numThreads; \
	ComputeGridSize(totalThreads, func ## _numBlocks, func ## _numThreads); \
	func <<<func ## _numBlocks, func ## _numThreads, stream>>>
#define CUDA_CALL_V(func, ...) \
	func <<<__VA_ARGS__>>>
#else
#define CUDA_CALL(func, totalThreads)
#define CUDA_CALL_S(func, totalThreads, stream) 
#define CUDA_CALL_V(func, ...)
#endif

namespace Velvet
{
	typedef unsigned int uint;
	// 定义块的大小为256
	const uint BLOCK_SIZE = 256;
	
	// 长度平方
	__device__ inline float length2(glm::vec3 vec)
	{
		return glm::dot(vec, vec);
	}

	// 计算Grid大小
	inline void ComputeGridSize(const uint& n, uint& numBlocks, uint& numThreads)
	{
		if (n == 0)
		{
			//fmt::print("Error(Solver): numParticles is 0\n");
			//printf()
			numBlocks = 0;
			numThreads = 0;
			return;
		}
		numThreads = min(n, BLOCK_SIZE);
		numBlocks = (n % numThreads != 0) ? (n / numThreads + 1) : (n / numThreads);
	}
	// 在CUDA上分配空间  返回分配空间的指针
	template<class T>
	inline T* VtAllocBuffer(size_t elementCount)
	{
		T* devPtr = nullptr;
		// 在CUDA端分配elementCount个元素空间
		checkCudaErrors(cudaMallocManaged((void**)&devPtr, elementCount * sizeof(T)));
		// 内存分配函数是异步执行的，当调用 cudaMallocManaged时，它会立即返回，而不会等待分配完成
		// cudaDeviceSynchronize 用于同步CUDA设备，确保之前的CUDA函数调用完成，包括内存分配操作
		cudaDeviceSynchronize(); // this is necessary, otherwise realloc can cause crash
		return devPtr;
	}

	// 释放内存
	inline void VtFreeBuffer(void* buffer)
	{
		checkCudaErrors(cudaFree(buffer));
	}
}