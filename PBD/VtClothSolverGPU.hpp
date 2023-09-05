#pragma once

#include <iostream>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include "cuda/helper_cuda.h"
#include "Mesh.hpp"
#include "VtClothSolverGPU.cuh"
#include "VtBuffer.hpp"
#include "SpatialHashGPU.hpp"
#include "MouseGrabber.hpp"
#include "Collider.hpp"

using namespace std;

namespace Velvet
{
	class VtClothSolverGPU : public Component
	{
	public:

		void Start() override
		{
			// ������Ϊ0
			Global::simParams.numParticles = 0;
			// �ҵ����е���ײ���
			m_colliders = Global::game->FindComponents<Collider>();
			// ��ʼ�����ץȡ
			m_mouseGrabber.Initialize(&positions, &velocities, &invMasses);
			//ShowDebugGUI();
		}

		void Update() override
		{
			// �������ץȡ����
			m_mouseGrabber.HandleMouseInteraction();
		}

		void FixedUpdate() override
		{
			// ����ץȡ����Ϣ(λ�á��ٶ�)
			m_mouseGrabber.UpdateGrappedVertex();
			// ������ײ
			UpdateColliders(m_colliders);

			Timer::StartTimer("GPU_TIME");
			// ģ��
			Simulate();
			Timer::EndTimer("GPU_TIME");
		}

		void OnDestroy() override
		{
			positions.destroy();
			normals.destroy();
		}

		void Simulate()
		{
			Timer::StartTimerGPU("Solver_Total");
			//==========================
			// Prepare
			//==========================
			// ֡ʱ��
			float frameTime = Timer::fixedDeltaTime();
			// ÿ֡ ÿ�� ������Ҫ��ʱ��
			float substepTime = Timer::fixedDeltaTime() / Global::simParams.numSubsteps;

			//==========================
			// Launch kernel
			//==========================
			// ����ģ�����
			SetSimulationParams(&Global::simParams);

			// External colliders can move relatively fast, and cloth will have large velocity after colliding with them.
			// This can produce unstable behavior, such as vertex flashing between two sides.
			// We include a pre-stabilization step to mitigate this issue. Collision here will not influence velocity.
			// ��һ��positions����ײ������   �ڶ�����ԭʼλ��
			// ��ײ��ÿ֡������ײ   ��������Ħ����
			CollideSDF(positions, sdfColliders, positions, (uint)sdfColliders.size(), frameTime);
			// Ԥ����ÿ֡ʱ����Բ���֮��Ĵ������е���
			for (int substep = 0; substep < Global::simParams.numSubsteps; ++substep)
			{
				// ��������
				PredictPositions(predicted, velocities, positions, substepTime);
				// �������ཻ
				if (Global::simParams.enableSelfCollision)
				{
					if (substep % Global::simParams.interleavedHash == 0)
					{
						m_spatialHash->Hash(predicted);
					}
					// ������ײ   
					CollideParticles(deltas, deltaCounts, predicted, invMasses, m_spatialHash->neighbors, positions);
				}
				// ÿ������ײ
				CollideSDF(predicted, sdfColliders, positions, (uint)sdfColliders.size(), substepTime);

				// ������Լ��
				for (int iteration = 0; iteration < Global::simParams.numIterations; iteration++)
				{

					// ����Լ��
					SolveStretch(predicted, deltas, deltaCounts, stretchIndices, stretchLengths, invMasses, (uint)stretchLengths.size());
					// ����Լ��
					SolveAttachment(predicted, deltas, deltaCounts, invMasses,
						attachParticleIDs, attachSlotIDs, attachSlotPositions, attachDistances, (uint)attachParticleIDs.size());
					// ����Լ��
					SolveBending(predicted, deltas, deltaCounts, bendIndices, bendAngles, invMasses, (uint)bendAngles.size(), substepTime);
					// Ӧ������
					ApplyDeltas(predicted, deltas, deltaCounts);
				}
				// ���յ���
				Finalize(velocities, positions, predicted, substepTime);
			}
			// ������ķ���
			ComputeNormal(normals, positions, indices, (uint)(indices.size() / 3));

			//==========================
			// ͬ������
			//==========================
			Timer::EndTimerGPU("Solver_Total");
			cudaDeviceSynchronize();

			// λ��ͬ��
			positions.sync();
			// ����ͬ��
			normals.sync();
		}
	public:

		int AddCloth(shared_ptr<Mesh> mesh, glm::mat4 modelMatrix, float particleDiameter)
		{
			Timer::StartTimer("INIT_SOLVER_GPU");
			// ������
			int prevNumParticles = Global::simParams.numParticles;
			// ����������
			int newParticles = (int)mesh->vertices().size();

			// Set global parameters
			// ����ȫ�ֲ���
			Global::simParams.numParticles += newParticles;
			Global::simParams.particleDiameter = particleDiameter;
			Global::simParams.deltaTime = Timer::fixedDeltaTime();
			Global::simParams.maxSpeed = 2 * particleDiameter / Timer::fixedDeltaTime() * Global::simParams.numSubsteps;

			// Allocate managed buffers
			// ����Buffer
			positions.registerNewBuffer(mesh->verticesVBO());
			normals.registerNewBuffer(mesh->normalsVBO());

			for (int i = 0; i < mesh->indices().size(); i++)
			{
				indices.push_back(mesh->indices()[i] + prevNumParticles);
			}

			velocities.push_back(newParticles, glm::vec3(0));
			predicted.push_back(newParticles, glm::vec3(0));
			deltas.push_back(newParticles, glm::vec3(0));
			deltaCounts.push_back(newParticles, 0);
			invMasses.push_back(newParticles, 1.0f);

			// ��ʼ��λ��
			InitializePositions(positions, prevNumParticles, newParticles, modelMatrix);
			cudaDeviceSynchronize();
			positions.sync();

			// ��ʼ����Ա����
			m_spatialHash = make_shared<SpatialHashGPU>(particleDiameter, Global::simParams.numParticles);
			m_spatialHash->SetInitialPositions(positions);

			double time = Timer::EndTimer("INIT_SOLVER_GPU") * 1000;

			//fmt::print("Info(ClothSolverGPU): AddCloth done. Took time {:.2f} ms\n", time);
			//fmt::print("Info(ClothSolverGPU): Use recommond max vel = {}\n", Global::simParams.maxSpeed);

			std::cout << "Info(ClothSolverGPU): AddCloth done. Took time " << std::fixed << std::setprecision(2) << time << " ms\n";
			std::cout << "Info(ClothSolverGPU): Use recommond max vel = " << Global::simParams.maxSpeed << "\n";

			return prevNumParticles;
		}

		// ��������Լ��
		void AddStretch(int idx1, int idx2, float distance)
		{
			// Լ������
			stretchIndices.push_back(idx1);
			stretchIndices.push_back(idx2);
			// Լ������
			stretchLengths.push_back(distance);
		}

		// ���Ӹ�����Լ��
		void AddAttachSlot(glm::vec3 attachSlotPos)
		{
			attachSlotPositions.push_back(attachSlotPos);
		}

		// ���Ӹ�����
		void AddAttach(int particleIndex, int slotIndex, float distance)
		{
			// ����Ϊ0��ʾ��ֹ
			if (distance == 0)
				invMasses[particleIndex] = 0;
			// ����������ID
			attachParticleIDs.push_back(particleIndex);
			// ���Ӳ�ID 
			attachSlotIDs.push_back(slotIndex);
			// ����
			attachDistances.push_back(distance);
		}

		// ��������Լ��
		void AddBend(uint idx1, uint idx2, uint idx3, uint idx4,float angle)
		{
			// ���������
			bendIndices.push_back(idx1);
			bendIndices.push_back(idx2);
			bendIndices.push_back(idx3);
			bendIndices.push_back(idx4);
			// ����ǽǶ�
			bendAngles.push_back(angle);
		}

		// ������ײ
		void UpdateColliders(vector<Collider*>& colliders)
		{
			// ��С����
			sdfColliders.resize(colliders.size());
			// ������ֵ
			for (int i = 0; i < colliders.size(); i++)
			{
				const Collider* c = colliders[i];
				if (!c->enabled) continue;
				SDFCollider sc;
				sc.type = c->type;
				sc.position = c->actor->transform->position;
				sc.scale = c->actor->transform->scale;
				sc.curTransform = c->curTransform;
				sc.invCurTransform = glm::inverse(c->curTransform);
				sc.lastTransform = c->lastTransform;
				sc.deltaTime = Timer::fixedDeltaTime();
				sdfColliders[i] = sc;
			}
		}

	public: // Sim buffers

		// positions��normals����Ҫ��OpenGL�����ģ�������VtMergedBuffer����
		VtMergedBuffer<glm::vec3> positions;
		VtMergedBuffer<glm::vec3> normals;
		VtBuffer<uint> indices;

		VtBuffer<glm::vec3> velocities;
		VtBuffer<glm::vec3> predicted;
		VtBuffer<glm::vec3> deltas;
		VtBuffer<int> deltaCounts;
		VtBuffer<float> invMasses;

		VtBuffer<int> stretchIndices;
		VtBuffer<float> stretchLengths;
		VtBuffer<uint> bendIndices;
		VtBuffer<float> bendAngles;

		// Attach attachParticleIndices[i] with attachSlotIndices[i] w
		// where their expected distance is attachDistances[i]
		VtBuffer<int> attachParticleIDs;
		VtBuffer<int> attachSlotIDs;
		VtBuffer<float> attachDistances;
		VtBuffer<glm::vec3> attachSlotPositions;

		VtBuffer<SDFCollider> sdfColliders;

	private:

		shared_ptr<SpatialHashGPU> m_spatialHash;
		vector<Collider*> m_colliders;
		MouseGrabber m_mouseGrabber;

		void ShowDebugGUI()
		{
			GUI::RegisterDebug([this]() {
				{
					static int particleIndex1 = 0;
					//IMGUI_LEFT_LABEL(ImGui::InputInt, "ParticleID", &particleIndex, 0, m_numParticles-1);
					IMGUI_LEFT_LABEL(ImGui::SliderInt, "ParticleID1", &particleIndex1, 0, Global::simParams.numParticles - 1);
					ImGui::Indent(10);
					//ImGui::Text(fmt::format("Position: {}", predicted[particleIndex1]).c_str());

					std::string text = "Position: " + Helper::to_string(predicted[particleIndex1]);
					ImGui::Text(text.c_str());

					auto hash3i = m_spatialHash->HashPosition3i(predicted[particleIndex1]);
					auto hash = m_spatialHash->HashPosition(predicted[particleIndex1]);

					//ImGui::Text(fmt::format("Hash: {}[{},{},{}]", hash, hash3i.x, hash3i.y, hash3i.z).c_str());
					std::ostringstream oss;
					oss << "Hash: " << hash << "[" << hash3i.x << "," << hash3i.y << "," << hash3i.z << "]";
					ImGui::Text(oss.str().c_str());

					auto norm = normals[particleIndex1];

					//ImGui::Text(fmt::format("Normal: [{:.3f},{:.3f},{:.3f}]", norm.x, norm.y, norm.z).c_str());
					ImGui::Text("Normal: [%.3f, %.3f, %.3f]", norm.x, norm.y, norm.z);

					static int neighborRange1 = 0;
					IMGUI_LEFT_LABEL(ImGui::SliderInt, "NeighborRange1", &neighborRange1, 0, 63);
					//ImGui::Text(fmt::format("NeighborID: {}", m_spatialHash->neighbors[neighborRange1 + particleIndex1 * Global::simParams.maxNumNeighbors]).c_str());
					ImGui::Text("NeighborID: %d", m_spatialHash->neighbors[neighborRange1 + particleIndex1 * Global::simParams.maxNumNeighbors]);
					ImGui::Indent(-10);
				}

				{
					static int particleIndex2 = 0;
					//IMGUI_LEFT_LABEL(ImGui::InputInt, "ParticleID", &particleIndex, 0, m_numParticles-1);
					IMGUI_LEFT_LABEL(ImGui::SliderInt, "ParticleID2", &particleIndex2, 0, Global::simParams.numParticles - 1);
					ImGui::Indent(10);
					//ImGui::Text(fmt::format("Position: {}", predicted[particleIndex2]).c_str());
					std::string text1 = "Position: " + Helper::to_string(predicted[particleIndex2]);
					ImGui::Text(text1.c_str());


					auto hash3i = m_spatialHash->HashPosition3i(predicted[particleIndex2]);
					auto hash = m_spatialHash->HashPosition(predicted[particleIndex2]);
					//ImGui::Text(fmt::format("Hash: {}[{},{},{}]", hash, hash3i.x, hash3i.y, hash3i.z).c_str());
					std::ostringstream oss;
					oss << "Hash: " << hash << "[" << hash3i.x << "," << hash3i.y << "," << hash3i.z << "]";
					ImGui::Text(oss.str().c_str());

					static int neighborRange2 = 0;
					IMGUI_LEFT_LABEL(ImGui::SliderInt, "NeighborRange2", &neighborRange2, 0, 63);
					//ImGui::Text(fmt::format("NeighborID: {}", m_spatialHash->neighbors[neighborRange2 + particleIndex2 * Global::simParams.maxNumNeighbors]).c_str());

					ImGui::Text("NeighborID: %d", m_spatialHash->neighbors[neighborRange2 + particleIndex2 * Global::simParams.maxNumNeighbors]);

					ImGui::Indent(-10);
				}
				static int cellID = 0;
				IMGUI_LEFT_LABEL(ImGui::SliderInt, "CellID", &cellID, 0, (int)m_spatialHash->cellStart.size() - 1);
				int start = m_spatialHash->cellStart[cellID];
				int end = m_spatialHash->cellEnd[cellID];
				ImGui::Indent(10);
				/*ImGui::Text(fmt::format("CellStart.HashID: {}", start).c_str());
				ImGui::Text(fmt::format("CellEnd.HashID: {}", end).c_str());*/
				ImGui::Text("CellStart.HashID: %d", start);
				ImGui::Text("CellEnd.HashID: %d", end);
				if (start != 0xffffffff && end > start)
				{
					static int particleHash = 0;
					particleHash = clamp(particleHash, start, end - 1);
					IMGUI_LEFT_LABEL(ImGui::SliderInt, "HashID", &particleHash, start, end - 1);

					std::string text = "ParticleHash: " + m_spatialHash->particleHash[particleHash];
					ImGui::Text(text.c_str());
					std::string text2 = "ParticleIndex: " + m_spatialHash->particleIndex[particleHash];
					ImGui::Text(text2.c_str());
				}
				});
		}
	};
}