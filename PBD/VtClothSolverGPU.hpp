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
			// 粒子数为0
			Global::simParams.numParticles = 0;
			// 找到所有的碰撞组件
			m_colliders = Global::game->FindComponents<Collider>();
			// 初始化鼠标抓取
			m_mouseGrabber.Initialize(&positions, &velocities, &invMasses);
			//ShowDebugGUI();
		}

		void Update() override
		{
			// 处理鼠标抓取交互
			m_mouseGrabber.HandleMouseInteraction();
		}

		void FixedUpdate() override
		{
			// 更新抓取点信息(位置、速度)
			m_mouseGrabber.UpdateGrappedVertex();
			// 处理碰撞
			UpdateColliders(m_colliders);

			Timer::StartTimer("GPU_TIME");
			// 模拟
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
			// 帧时间
			float frameTime = Timer::fixedDeltaTime();
			// 每帧 每次 迭代需要的时间
			float substepTime = Timer::fixedDeltaTime() / Global::simParams.numSubsteps;

			//==========================
			// Launch kernel
			//==========================
			// 设置模拟参数
			SetSimulationParams(&Global::simParams);

			// External colliders can move relatively fast, and cloth will have large velocity after colliding with them.
			// This can produce unstable behavior, such as vertex flashing between two sides.
			// We include a pre-stabilization step to mitigate this issue. Collision here will not influence velocity.
			// 第一个positions是碰撞修正后   第二个是原始位置
			// 碰撞是每帧进行碰撞   ，计算了摩擦力
			CollideSDF(positions, sdfColliders, positions, (uint)sdfColliders.size(), frameTime);
			// 预测是每帧时间除以步长之后的次数进行迭代
			for (int substep = 0; substep < Global::simParams.numSubsteps; ++substep)
			{
				// 重力修正
				PredictPositions(predicted, velocities, positions, substepTime);
				// 处理自相交
				if (Global::simParams.enableSelfCollision)
				{
					if (substep % Global::simParams.interleavedHash == 0)
					{
						m_spatialHash->Hash(predicted);
					}
					// 粒子碰撞   
					CollideParticles(deltas, deltaCounts, predicted, invMasses, m_spatialHash->neighbors, positions);
				}
				// 每步长碰撞
				CollideSDF(predicted, sdfColliders, positions, (uint)sdfColliders.size(), substepTime);

				// 迭代求约束
				for (int iteration = 0; iteration < Global::simParams.numIterations; iteration++)
				{

					// 拉伸约束
					SolveStretch(predicted, deltas, deltaCounts, stretchIndices, stretchLengths, invMasses, (uint)stretchLengths.size());
					// 附加约束
					SolveAttachment(predicted, deltas, deltaCounts, invMasses,
						attachParticleIDs, attachSlotIDs, attachSlotPositions, attachDistances, (uint)attachParticleIDs.size());
					// 弯曲约束
					SolveBending(predicted, deltas, deltaCounts, bendIndices, bendAngles, invMasses, (uint)bendAngles.size(), substepTime);
					// 应用修正
					ApplyDeltas(predicted, deltas, deltaCounts);
				}
				// 最终叠加
				Finalize(velocities, positions, predicted, substepTime);
			}
			// 修正后的法线
			ComputeNormal(normals, positions, indices, (uint)(indices.size() / 3));

			//==========================
			// 同步操作
			//==========================
			Timer::EndTimerGPU("Solver_Total");
			cudaDeviceSynchronize();

			// 位置同步
			positions.sync();
			// 法线同步
			normals.sync();
		}
	public:

		int AddCloth(shared_ptr<Mesh> mesh, glm::mat4 modelMatrix, float particleDiameter)
		{
			Timer::StartTimer("INIT_SOLVER_GPU");
			// 粒子数
			int prevNumParticles = Global::simParams.numParticles;
			// 新粒子数量
			int newParticles = (int)mesh->vertices().size();

			// Set global parameters
			// 设置全局参数
			Global::simParams.numParticles += newParticles;
			Global::simParams.particleDiameter = particleDiameter;
			Global::simParams.deltaTime = Timer::fixedDeltaTime();
			Global::simParams.maxSpeed = 2 * particleDiameter / Timer::fixedDeltaTime() * Global::simParams.numSubsteps;

			// Allocate managed buffers
			// 分配Buffer
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

			// 初始化位置
			InitializePositions(positions, prevNumParticles, newParticles, modelMatrix);
			cudaDeviceSynchronize();
			positions.sync();

			// 初始化成员变量
			m_spatialHash = make_shared<SpatialHashGPU>(particleDiameter, Global::simParams.numParticles);
			m_spatialHash->SetInitialPositions(positions);

			double time = Timer::EndTimer("INIT_SOLVER_GPU") * 1000;

			//fmt::print("Info(ClothSolverGPU): AddCloth done. Took time {:.2f} ms\n", time);
			//fmt::print("Info(ClothSolverGPU): Use recommond max vel = {}\n", Global::simParams.maxSpeed);

			std::cout << "Info(ClothSolverGPU): AddCloth done. Took time " << std::fixed << std::setprecision(2) << time << " ms\n";
			std::cout << "Info(ClothSolverGPU): Use recommond max vel = " << Global::simParams.maxSpeed << "\n";

			return prevNumParticles;
		}

		// 增加拉伸约束
		void AddStretch(int idx1, int idx2, float distance)
		{
			// 约束索引
			stretchIndices.push_back(idx1);
			stretchIndices.push_back(idx2);
			// 约束距离
			stretchLengths.push_back(distance);
		}

		// 增加附加物约束
		void AddAttachSlot(glm::vec3 attachSlotPos)
		{
			attachSlotPositions.push_back(attachSlotPos);
		}

		// 增加附加物
		void AddAttach(int particleIndex, int slotIndex, float distance)
		{
			// 质量为0表示静止
			if (distance == 0)
				invMasses[particleIndex] = 0;
			// 附加物粒子ID
			attachParticleIDs.push_back(particleIndex);
			// 附加槽ID 
			attachSlotIDs.push_back(slotIndex);
			// 距离
			attachDistances.push_back(distance);
		}

		// 增加弯曲约束
		void AddBend(uint idx1, uint idx2, uint idx3, uint idx4,float angle)
		{
			// 二面角索引
			bendIndices.push_back(idx1);
			bendIndices.push_back(idx2);
			bendIndices.push_back(idx3);
			bendIndices.push_back(idx4);
			// 二面角角度
			bendAngles.push_back(angle);
		}

		// 更新碰撞
		void UpdateColliders(vector<Collider*>& colliders)
		{
			// 大小更新
			sdfColliders.resize(colliders.size());
			// 挨个赋值
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

		// positions和normals是需要和OpenGL交互的，所以是VtMergedBuffer类型
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