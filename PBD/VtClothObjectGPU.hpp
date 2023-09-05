#pragma once

#include "Component.hpp"
#include "VtClothSolverGPU.hpp"
#include "Actor.hpp"
#include "MeshRenderer.hpp"
#include "VtEngine.hpp"
#include "VtClothSolverGPU.hpp"


namespace Velvet
{
	class VtClothObjectGPU : public Component
	{
	public:
		VtClothObjectGPU(int resolution, shared_ptr<VtClothSolverGPU> solver)
		{
			SET_COMPONENT_NAME;

			m_solver = solver;
			m_resolution = resolution;
		}

		// indices决定了某个顶点的附着状态
		/*
		indices表示某个顶点的索引，0表示第0个顶点，15表示第15个顶点
		*/
		void SetAttachedIndices(vector<int> indices)
		{
			m_attachedIndices = indices;
		}

		// 粒子直径
		auto particleDiameter() const
		{
			return m_particleDiameter;
		}

		auto solver() const
		{
			return m_solver;
		}

		VtBuffer<glm::vec3>& attachSlotPositions() const
		{
			return m_solver->attachSlotPositions;
		}

	public:
		void Start() override
		{
			// 获取网格
			auto mesh = actor->GetComponent<MeshRenderer>()->mesh();
			// 获取 transform
			auto transformMatrix = actor->transform->matrix();
			// 位置
			auto positions = mesh->vertices();
			// 索引
			auto indices = mesh->indices();
			m_particleDiameter = glm::length(positions[0] - positions[1]) * Global::simParams.particleDiameterScalar;

			//std::cout << Helper::to_string(positions[0] - positions[1]) << std::endl;

			m_indexOffset = m_solver->AddCloth(mesh, transformMatrix, m_particleDiameter);

			std::cout << "m_indexOffset" << m_indexOffset << std::endl;
			actor->transform->Reset();

			ApplyTransform(positions, transformMatrix);

			GenerateStretch(positions);
			//GenerateAttach(positions);
			GenerateBending(indices,positions);
		}

	private:
		int m_resolution;
		int m_indexOffset;
		shared_ptr<VtClothSolverGPU> m_solver;
		vector<int> m_attachedIndices;
		float m_particleDiameter;

		// 应用Transform矩阵（这个矩阵囊括了位置、旋转、缩放）
		void ApplyTransform(vector<glm::vec3>& positions, glm::mat4 transform)
		{
			for (int i = 0; i < positions.size(); i++)
			{
				positions[i] = transform * glm::vec4(positions[i], 1.0);
			}
		}

		// 布料生成拉伸约束原本
//void GenerateStretch(const vector<glm::vec3>& positions)
//		{
//			auto VertexAt = [this](int x, int y) {
//				return x * (m_resolution + 1) + y;
//			};
//			auto DistanceBetween = [&positions](int idx1, int idx2) {
//				return glm::length(positions[idx1] - positions[idx2]);
//			};
//
//			for (int x = 0; x < m_resolution + 1; x++)
//			{
//				for (int y = 0; y < m_resolution + 1; y++)
//				{
//					int idx1, idx2;
//
//					if (y != m_resolution)
//					{
//						idx1 = VertexAt(x, y);
//						idx2 = VertexAt(x, y + 1);
//						m_solver->AddStretch(m_indexOffset + idx1, m_indexOffset + idx2, DistanceBetween(idx1, idx2));
//					}
//
//					if (x != m_resolution)
//					{
//						idx1 = VertexAt(x, y);
//						idx2 = VertexAt(x + 1, y);
//						m_solver->AddStretch(m_indexOffset + idx1, m_indexOffset + idx2, DistanceBetween(idx1, idx2));
//					}
//
//					if (y != m_resolution && x != m_resolution)
//					{
//						idx1 = VertexAt(x, y);
//						idx2 = VertexAt(x + 1, y + 1);
//						m_solver->AddStretch(m_indexOffset + idx1, m_indexOffset + idx2, DistanceBetween(idx1, idx2));
//
//						idx1 = VertexAt(x, y + 1);
//						idx2 = VertexAt(x + 1, y);
//						m_solver->AddStretch(m_indexOffset + idx1, m_indexOffset + idx2, DistanceBetween(idx1, idx2));
//					}
//				}
//			}
//		}
		//直愣愣
		void GenerateStretch(const vector<glm::vec3>& positions)
		{
			auto VertexAt = [this](int x, int y) {
				return x * (m_resolution + 1) + y;
			};
			auto DistanceBetween = [&positions](int idx1, int idx2) {
				return glm::length(positions[idx1] - positions[idx2]);
			};
			float defaultMaxStretchDistance = 5.0f; // 设置默认的最大拉伸距离为5.0f

			for (int i = 0; i < positions.size(); i++)
			{
				for (int j = i + 1; j < positions.size(); j++)
				{
					glm::vec3 kk = positions[i] - positions[j];
					glm::vec3 n = glm::normalize(kk);
					float distance = glm::length(positions[i] - positions[j]);
					if (distance <= defaultMaxStretchDistance)
					{
						//添加拉伸约束
						m_solver->AddStretch(i, j, distance);
					}
				}
			}
		}
		//void GenerateStretch(const vector<glm::vec3>& positions)
		//{
		//	float defaultMaxStretchDistance = 20.0f; // 设置默认的最大拉伸距离为5.0f
		//	for (int i = 0; i < positions.size(); i++)
		//	{
		//		for (int j = i + 1; j < positions.size(); j++)
		//		{
		//			float distance = glm::length(positions[i] - positions[j]);
		//			if (distance <= defaultMaxStretchDistance)
		//			{
		//				// 添加拉伸约束
		//				m_solver->AddStretch(i, j, distance);
		//			}
		//		}
		//	}
		//}
		// 绳子生成拉伸约束
		//void GenerateStretch(const vector<glm::vec3>& positions)
		//{
		//	float defaultMaxStretchDistance = 20.0f; // 设置默认的最大拉伸距离为20.0f
		//	int vertLineNum = positions.size() / 6; // 六面体的顶点行数
		//	for (int i = 0; i < vertLineNum; i++)
		//	{
		//		for (int j = 0; j < 6; j++)
		//		{
		//			int vertIndex = i * 6 + j;
		//			// 获取当前顶点的索引
		//			int currentIndex = vertIndex;
		//			// 获取相邻顶点的索引
		//			int nextIndex = (j == 5) ? (i * 6) : (vertIndex + 1);
		//			int downIndex = (i == vertLineNum - 1) ? j : (vertIndex + 6);
		//			// 获取当前顶点和相邻顶点的位置
		//			const glm::vec3& currentPosition = positions[currentIndex];
		//			const glm::vec3& nextPosition = positions[nextIndex];
		//			const glm::vec3& downPosition = positions[downIndex];
		//			// 计算当前顶点和相邻顶点的距离
		//			float currentNextDistance = glm::length(currentPosition - nextPosition);
		//			float currentDownDistance = glm::length(currentPosition - downPosition);
		//		//	 添加拉伸约束
		//			if (currentNextDistance <= defaultMaxStretchDistance)
		//			{
		//				m_solver->AddStretch(currentIndex, nextIndex, currentNextDistance);
		//			}
		//			if (currentDownDistance <= defaultMaxStretchDistance)
		//			{
		//				m_solver->AddStretch(currentIndex, downIndex, currentDownDistance);
		//			}
		//		}
		//	}
		//}

		 //void GenerateStretch(const vector<glm::vec3>& positions)
		//{
		//	auto VertexAt = [](int x, int y, int resolution) {
		//		return x * (resolution + 1) + y;
		//	};
		//	auto DistanceBetween = [&positions](int idx1, int idx2) {
		//		return glm::length(positions[idx1] - positions[idx2]);
		//	};
		//	int resolution = static_cast<int>(sqrt(positions.size())) - 1;  // 计算六边形网格的分辨率
		//	for (int x = 0; x < resolution + 1; x++)
		//	{
		//		for (int y = 0; y < resolution + 1; y++)
		//		{
		//			int idx1, idx2;
		//			if (y != resolution)
		//			{
		//				idx1 = VertexAt(x, y, resolution);
		//				idx2 = VertexAt(x, y + 1, resolution);
		//				m_solver->AddStretch(idx1, idx2, DistanceBetween(idx1, idx2));
		//			}
		//			if (x != resolution)
		//			{
		//				idx1 = VertexAt(x, y, resolution);
		//				idx2 = VertexAt(x + 1, y, resolution);
		//				m_solver->AddStretch(idx1, idx2, DistanceBetween(idx1, idx2));
		//			}
		//			if (y != resolution && x != resolution)
		//			{
		//				idx1 = VertexAt(x, y, resolution);
		//				idx2 = VertexAt(x + 1, y + 1, resolution);
		//				m_solver->AddStretch(idx1, idx2, DistanceBetween(idx1, idx2));
		//				idx1 = VertexAt(x, y + 1, resolution);
		//				idx2 = VertexAt(x + 1, y, resolution);
		//				m_solver->AddStretch(idx1, idx2, DistanceBetween(idx1, idx2));
		//			}
		//		}
		//	}
		//}
		// 产生弯曲约束
		void GenerateBending(const vector<unsigned int>& indices,const vector<glm::vec3>& positions)
		{
			// HACK: not for every kind of mesh
			for (int i = 0; i < indices.size(); i += 6)
			{
				int idx1 = indices[i];
				int idx2 = indices[i + 1];
				int idx3 = indices[i + 2];
				int idx4 = indices[i + 5];

				// TODO: calculate angle
				glm::vec3 edge1 = positions[m_indexOffset + idx2] - positions[m_indexOffset + idx1];
				glm::vec3 edge2 = positions[m_indexOffset + idx4] - positions[m_indexOffset + idx3];
				float angle = glm::acos(glm::dot(glm::normalize(edge1), glm::normalize(edge2))) * 180.0f / 3.14159265359f;
			//	float angle = 180;
				m_solver->AddBend(m_indexOffset + idx1, m_indexOffset + idx2, m_indexOffset + idx3, m_indexOffset + idx4, angle);
			}
		}


		// 产生附着
		//void GenerateAttach(const vector<glm::vec3>& positions)
		//{
		//	for (int slotIdx = 0; slotIdx < m_attachedIndices.size(); slotIdx++)
		//	{
		//		// 粒子IDs
		//		int particleID = m_attachedIndices[slotIdx];
		//		// 粒子ID的位置
		//		glm::vec3 slotPos = positions[particleID];
		//		m_solver->AddAttachSlot(slotPos);
		//		for (int i = 0; i < positions.size(); i++)
		//		{
		//			float restDistance = glm::length(slotPos - positions[i]);
		//			m_solver->AddAttach(m_indexOffset + i, slotIdx, restDistance);
		//		}
		//		m_solver->AddAttach(idx, positions[idx], 0);
		//	}
		//}
		//void GenerateAttach(const vector<glm::vec3>& positions)
		//{
		//	int numParticles = positions.size();
		//	int numSlots = numParticles / 8; // 六边形网格的附着约束每个六边形有8个顶点
		//	for (int slotIdx = 0; slotIdx < numSlots; slotIdx++)
		//	{
		//		// 计算六边形网格的中心点位置
		//		glm::vec3 slotPos(0.0f);
		//		for (int i = 0; i < 8; i++)
		//		{
		//			int particleID = slotIdx * 8 + i;
		//			slotPos += positions[particleID];
		//		}
		//		slotPos /= 8.0f;
		//		m_solver->AddAttachSlot(slotPos);
		//		// 添加附着约束
		//		for (int i = 0; i < numParticles; i++)
		//		{
		//			float restDistance = glm::length(slotPos - positions[i]);
		//			m_solver->AddAttach(i, slotIdx, restDistance);
		//		}
		//	}
		//}
		void GenerateAttach(const vector<glm::vec3>& positions)
		{
			int numParticles = positions.size();
			int numSlots = numParticles / 8; // 六边形网格的附着约束每个六边形有8个顶点

			for (int slotIdx = 0; slotIdx < numSlots; slotIdx++)
			{
				glm::vec3 slotPos = positions[slotIdx * 8]; // 使用绳子上第一个粒子的位置作为附着点位置

				m_solver->AddAttachSlot(slotPos);
				// 添加附着约束
				for (int i = 0; i < numParticles; i++)
				{
					float restDistance = glm::length(slotPos - positions[i]);
					m_solver->AddAttach(i, slotIdx, restDistance);
				}
			}
		}
	};
}