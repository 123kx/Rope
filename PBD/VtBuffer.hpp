#pragma once

#include "Common.cuh"

namespace Velvet
{
	/*
	存储T类型的数据的数据结构，它类似于Vector,但是它的内存是在CUDA设备端分配的
	*/
	template <class T>
	class VtBuffer
	{
	public:
		// 无参构造
		VtBuffer() {}
		// 大小参数的构造
		VtBuffer(uint size)
		{
			resize(size);
		}
		// 
		VtBuffer(const VtBuffer&) = delete;

		VtBuffer& operator=(const VtBuffer&) = delete;

		operator T* () const { return m_buffer; }

		~VtBuffer()
		{
			destroy();
		}

		T& operator[](size_t index)
		{
			assert(m_buffer);
			assert(index < m_count);
			return m_buffer[index];
		}

		size_t size() const { return m_count; }

		void push_back(const T& t)
		{
			reserve(m_count + 1);
			m_buffer[m_count++] = t;
		}

		void push_back(size_t newCount, const T& val)
		{
			for (int i = 0; i < newCount; i++)
			{
				push_back(val);
			}
		}

		void push_back(const vector<T>& data)
		{
			size_t offset = m_count;
			resize(m_count + data.size());
			memcpy(m_buffer + offset, data.data(), data.size() * sizeof(T));
		}

		// 扩容
		void reserve(size_t minCapacity)
		{
			if (minCapacity > m_capacity)
			{
				// 每次增长1.5倍
				const size_t newCapacity = minCapacity * 3 / 2;

				T* newBuf = VtAllocBuffer<T>(newCapacity);

				// copy contents to new buffer			
				if (m_buffer)
				{
					// cudaMallocManaged分配的内存，可以在主机和设备之间进行透明访问
					// 意味着可以在主机上使用标准的内存操作函数来复制数据到CUDA管理的内存，
					// 以及从CUDA管理的内存中复制数据回主机
					memcpy(newBuf, m_buffer, m_count * sizeof(T));
					// 拷贝完，释放内存
					VtFreeBuffer(m_buffer);
				}

				// 交换
				m_buffer = newBuf;
				m_capacity = newCapacity;
			}
		}

		void resize(size_t newCount)
		{
			reserve(newCount);
			m_count = newCount;
		}

		void resize(size_t newCount, const T& val)
		{
			const size_t startInit = m_count;
			const size_t endInit = newCount;

			resize(newCount);

			// init any new entries
			for (size_t i = startInit; i < endInit; ++i)
				m_buffer[i] = val;
		}

		T* data() const
		{
			return m_buffer;
		}

		// 销毁
		void destroy()
		{
			if (m_buffer != nullptr)
			{
				VtFreeBuffer(m_buffer);
			}
			m_count = 0;
			m_capacity = 0;
			m_buffer = nullptr;
		}

	private:
		size_t m_count = 0;
		size_t m_capacity = 0;
		T* m_buffer = nullptr;
	};

	/*
	CUDA提供了与OpenGL交互功能，可以使用CUDA进行计算并将结果直接用于OpenGL渲染
	*/
	template <class T>
	class VtRegisteredBuffer
	{
	public:
		VtRegisteredBuffer() {}

		VtRegisteredBuffer(const VtRegisteredBuffer&) = delete;

		VtRegisteredBuffer& operator=(const VtRegisteredBuffer&) = delete;

		~VtRegisteredBuffer()
		{
			destroy();
		}

		T* data() const { return m_buffer; }

		operator T* () const { return m_buffer; }

		T& operator[](size_t index)
		{
			assert(m_bufferCPU);
			assert(index < m_count);
			return m_bufferCPU[index];
		}

		size_t size() const { return m_count; }

		void destroy()
		{
			if (m_cudaVboResource != nullptr)
			{
				//fmt::print("Info(VtBuffer): Release CUDA Resource ({})\n", (int)m_cudaVboResource);
				checkCudaErrors(cudaGraphicsUnregisterResource(m_cudaVboResource));
			}
			if (m_bufferCPU)
			{
				cudaFree(m_bufferCPU);
			}
			m_count = 0;
			m_buffer = nullptr;
			m_bufferCPU = nullptr;
			m_cudaVboResource = nullptr;
		}
	public:
		// CUDA interop with OpenGL
		void registerBuffer(GLuint vbo)
		{
			/* 
			参数一：指向cudaGraphicsResource指针的指针，用于存储注册后的CUDA图形资源，
					通过该指针，可以在后续的CUDA操作中访问和操作OpenGL缓冲区
			参数二：注册为CUDA图形资源的OpenGL缓冲区对象标识符
					它是通过OpenGL函数(glGenBuffers、glBindBuffer)分配和绑定的缓冲区对象
			参数三：可选的标志参数
			*/
			checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cudaVboResource, vbo, cudaGraphicsRegisterFlagsNone));

			// map (example 'gl_cuda_interop_pingpong_st' says map and unmap only needs to be done once)
			/*
			参数一：映射的CUDA图形资源的数量
			参数二：要映射的CUDA图形资源的数组
			参数三：可选参数，用于指定用于映射操作的CUDA流
			*/
			checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaVboResource, 0));
			
			/*
			参数一：输出参数，用于存储获取到的CUDA指针的指针，通过它，可以在CUDA内核中访问OpenGL缓冲区的数据
			参数二：输出参数，用于存储映射区域的大小的指针，可以使用该大小来确定映射区域的有效范围
			*/
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_buffer, &m_numBytes,
				m_cudaVboResource));
			m_count = m_numBytes / sizeof(T);

			/*
			参数一：取消映射的CUDA图形资源的数量
			参数二：要取消的CUDA图形资源的数组
			*/
			checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cudaVboResource, 0));
		}

		size_t m_count = 0;
		size_t m_numBytes = 0;
		T* m_buffer = nullptr;
		T* m_bufferCPU = nullptr;

		struct cudaGraphicsResource* m_cudaVboResource = nullptr;
	};

	template <class T>
	class VtMergedBuffer
	{
	public:
		VtMergedBuffer() {}
		VtMergedBuffer(const VtMergedBuffer&) = delete;
		VtMergedBuffer& operator=(const VtMergedBuffer&) = delete;

		void destroy()
		{
			m_vbuffer.destroy();
			m_rbuffers.clear();
		}

		void registerNewBuffer(GLuint vbo)
		{
			auto rbuf = make_shared<VtRegisteredBuffer<T>>();
			rbuf->registerBuffer(vbo);
			m_rbuffers.push_back(rbuf);

			size_t last = m_offsets.size() - 1;
			size_t offset = m_offsets.empty() ? 0 : m_offsets[last] + m_rbuffers[last]->size();
			m_offsets.push_back(offset);

			// copy from rbuffers to vbuffer
			m_vbuffer.resize(m_vbuffer.size() + rbuf->size());
			cudaMemcpy(m_vbuffer.data() + offset, rbuf->data(), rbuf->size() * sizeof(T), cudaMemcpyDefault);
		}

		size_t size() const
		{
			return m_vbuffer.size();
		}


		// 同步
		void sync()
		{
			// copy from vbuffer to rbuffers
			for (int i = 0; i < m_rbuffers.size(); i++)
			{
				cudaMemcpy(m_rbuffers[i]->data(), m_vbuffer.data() + m_offsets[i], m_rbuffers[i]->size() * sizeof(T), cudaMemcpyDefault);
			}
		}

		operator T* () const { return m_vbuffer.data(); }
	private:
		vector<shared_ptr<VtRegisteredBuffer<T>>> m_rbuffers;
		vector<size_t> m_offsets;
		VtBuffer<T> m_vbuffer;
	};
}