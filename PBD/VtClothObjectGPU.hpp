#pragma once

#include "Component.hpp"
#include "VtClothSolverGPU.hpp"
#include "Actor.hpp"
#include "MeshRenderer.hpp"
#include "VtEngine.hpp"

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

		VtBuffer<glm::vec3> &attachSlotPositions() const
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
			GenerateBending(indices);
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
	/*	void GenerateStretch(const vector<glm::vec3>& positions)
		{
			auto VertexAt = [this](int x, int y) {
				return x * (m_resolution + 1) + y;
			};
			auto DistanceBetween = [&positions](int idx1, int idx2) {
				return glm::length(positions[idx1] - positions[idx2]);
			};

			for (int x = 0; x < m_resolution + 1; x++)
			{
				for (int y = 0; y < m_resolution + 1; y++)
				{
					int idx1, idx2;

					if (y != m_resolution)
					{
						idx1 = VertexAt(x, y);
						idx2 = VertexAt(x, y + 1);
						m_solver->AddStretch(m_indexOffset + idx1, m_indexOffset + idx2, DistanceBetween(idx1, idx2));
					}

					if (x != m_resolution)
					{
						idx1 = VertexAt(x, y);
						idx2 = VertexAt(x + 1, y);
						m_solver->AddStretch(m_indexOffset + idx1, m_indexOffset + idx2, DistanceBetween(idx1, idx2));
					}

					if (y != m_resolution && x != m_resolution)
					{
						idx1 = VertexAt(x, y);
						idx2 = VertexAt(x + 1, y + 1);
						m_solver->AddStretch(m_indexOffset + idx1, m_indexOffset + idx2, DistanceBetween(idx1, idx2));

						idx1 = VertexAt(x, y + 1);
						idx2 = VertexAt(x + 1, y);
						m_solver->AddStretch(m_indexOffset + idx1, m_indexOffset + idx2, DistanceBetween(idx1, idx2));
					}
				}
			}
		}*/
		// 绳子生成拉伸约束
		void GenerateStretch(const vector<glm::vec3>& positions)
		{
			float defaultMaxStretchDistance = 20.0f; // 设置默认的最大拉伸距离为5.0f

			for (int i = 0; i < positions.size(); i++)
			{
				for (int j = i + 1; j < positions.size(); j++)
				{
					float distance = glm::length(positions[i] - positions[j]);
					if (distance <= defaultMaxStretchDistance)
					{
						// 添加拉伸约束
						m_solver->AddStretch(i, j, distance);
					}
				}
			}
		}
	

		// 产生弯曲约束
		void GenerateBending(const vector<unsigned int>& indices)
		{
			// HACK: not for every kind of mesh
			for (int i = 0; i < indices.size(); i += 6)
			{
				int idx1 = indices[i];
				int idx2 = indices[i + 1];
				int idx3 = indices[i + 2];
				int idx4 = indices[i + 5];

				// TODO: calculate angle
				float angle = 0;
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