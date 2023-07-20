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

		// indices������ĳ������ĸ���״̬
		/*
		indices��ʾĳ�������������0��ʾ��0�����㣬15��ʾ��15������
		*/
		void SetAttachedIndices(vector<int> indices)
		{
			m_attachedIndices = indices;
		}

		// ����ֱ��
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
			// ��ȡ����
			auto mesh = actor->GetComponent<MeshRenderer>()->mesh();
			// ��ȡ transform
			auto transformMatrix = actor->transform->matrix();
			// λ��
			auto positions = mesh->vertices();
			// ����
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

		// Ӧ��Transform�����������������λ�á���ת�����ţ�
		void ApplyTransform(vector<glm::vec3>& positions, glm::mat4 transform)
		{
			for (int i = 0; i < positions.size(); i++)
			{
				positions[i] = transform * glm::vec4(positions[i], 1.0);
			}
		}

		// ������������Լ��ԭ��
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
		// ������������Լ��
		void GenerateStretch(const vector<glm::vec3>& positions)
		{
			float defaultMaxStretchDistance = 20.0f; // ����Ĭ�ϵ�����������Ϊ5.0f

			for (int i = 0; i < positions.size(); i++)
			{
				for (int j = i + 1; j < positions.size(); j++)
				{
					float distance = glm::length(positions[i] - positions[j]);
					if (distance <= defaultMaxStretchDistance)
					{
						// �������Լ��
						m_solver->AddStretch(i, j, distance);
					}
				}
			}
		}
	

		// ��������Լ��
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
		 

		// ��������
		//void GenerateAttach(const vector<glm::vec3>& positions)
		//{
		//	for (int slotIdx = 0; slotIdx < m_attachedIndices.size(); slotIdx++)
		//	{
		//		// ����IDs
		//		int particleID = m_attachedIndices[slotIdx];
		//		// ����ID��λ��
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
		//	int numSlots = numParticles / 8; // ����������ĸ���Լ��ÿ����������8������

		//	for (int slotIdx = 0; slotIdx < numSlots; slotIdx++)
		//	{
		//		// ������������������ĵ�λ��
		//		glm::vec3 slotPos(0.0f);
		//		for (int i = 0; i < 8; i++)
		//		{
		//			int particleID = slotIdx * 8 + i;
		//			slotPos += positions[particleID];
		//		}
		//		slotPos /= 8.0f;

		//		m_solver->AddAttachSlot(slotPos);

		//		// ��Ӹ���Լ��
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
			int numSlots = numParticles / 8; // ����������ĸ���Լ��ÿ����������8������

			for (int slotIdx = 0; slotIdx < numSlots; slotIdx++)
			{
				glm::vec3 slotPos = positions[slotIdx * 8]; // ʹ�������ϵ�һ�����ӵ�λ����Ϊ���ŵ�λ��

				m_solver->AddAttachSlot(slotPos);

				// ��Ӹ���Լ��
				for (int i = 0; i < numParticles; i++)
				{
					float restDistance = glm::length(slotPos - positions[i]);
					m_solver->AddAttach(i, slotIdx, restDistance);
				}
			}
		}
	};
}