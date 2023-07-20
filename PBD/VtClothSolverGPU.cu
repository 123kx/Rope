#include "VtClothSolverGPU.cuh"
#include "Common.cuh"
#include "Common.hpp"
#include "Timer.hpp"

// ����ʹ�ã�������꣬�м�*****ע��*****


using namespace std;

namespace Velvet
{
	// constant ָ�������ڴ棬����GPU��������ķ��ʷ�ʽ�������ṩ����Ķ�ȡ����
	// ���ǣ���Щ����������CUDA�����ִ�й����У����ǲ����޸ĵģ������Ա���ȡ
	__device__ __constant__ VtSimParams d_params;
	VtSimParams h_params;

	__device__ inline void AtomicAdd(glm::vec3* address, int index, glm::vec3 val, int reorder)
	{
		// reorder��֤��ԭ�Ӳ�������ȷ��
		/*
		���Խ�����֤��ʹ����reorder֮��r1��r2��r3��ֵΨһ�����������������㣩
		*/
		int r1 = reorder % 3;
		int r2 = (reorder + 1) % 3;
		int r3 = (reorder + 2) % 3;
		// ʵ��������ֵ�ķ���������ۼӲ���
		atomicAdd(&(address[index].x) + r1, val[r1]);
		atomicAdd(&(address[index].x) + r2, val[r2]);
		atomicAdd(&(address[index].x) + r3, val[r3]);
	}

	// ����ģ�����ʱ��
	void SetSimulationParams(VtSimParams* hostParams)
	{
		ScopedTimerGPU timer("Solver_SetParams");
		// checkCudaErrors���CUDA�����ķ���ֵ�������κδ���
		// cudaMemcpyToSymbolAsync ���ڽ����ݴ������ڴ��첽���Ƶ��豸�ĳ����ڴ�
		/*
		�������ڴ��е�����ͨ���첽�������Ƶ��豸�ĳ����ڴ�"d_params"�У�
		�����豸�ϵĺ�������ʹ�ó����ڴ��е����ݽ��м��㣬������Ҫ�������ڴ��ȡ���ݣ��Ӷ��������
		*/
		checkCudaErrors(cudaMemcpyToSymbolAsync(d_params, hostParams, sizeof(VtSimParams)));
		h_params = *hostParams;
	}

	// ��ʼ��λ��
	__global__ void InitializePositions_Kernel(glm::vec3* positions, const int start, const int count, const glm::mat4 modelMatrix)
	{
		GET_CUDA_ID(id, count);
		// ��λ��ת��������������
		positions[start + id] = modelMatrix * glm::vec4(positions[start+id], 1);
	}

	void InitializePositions(glm::vec3* positions, const int start, const int count, const glm::mat4 modelMatrix)
	{
		ScopedTimerGPU timer("Solver_Initialize");
		CUDA_CALL(InitializePositions_Kernel, count)(positions, start, count, modelMatrix);
	}

	// Ԥ��λ��  ���������position�ǲ��ܸ��ĵģ�����ΪCONST
	__global__ void PredictPositions_Kernel(
		glm::vec3* predicted,
		glm::vec3* velocities,
		CONST(glm::vec3*) positions,
		const float deltaTime)
	{
		/*
		�ж��ٸ����ӣ��ʹ������ٸ��̣߳�
		*/
		GET_CUDA_ID(id, d_params.numParticles);

		//glm::vec3 gravity = glm::vec3(0, -10, 0);
		// �����ٶ�
		velocities[id] += d_params.gravity * deltaTime;
		// ����λ��
		predicted[id] = positions[id] + velocities[id] * deltaTime;
	}

	void PredictPositions(
		glm::vec3* predicted, 
		glm::vec3* velocities,
		CONST(glm::vec3*) positions,
		const float deltaTime)
	{
		ScopedTimerGPU timer("Solver_Predict");
		CUDA_CALL(PredictPositions_Kernel, h_params.numParticles)(predicted, velocities, positions, deltaTime);
	}

	// ��������Լ�� ( deltas��ʵ����delta_p )
	__global__ void SolveStretch_Kernel(
		glm::vec3* predicted,
		glm::vec3* deltas,
		int* deltaCounts,
		CONST(int*) stretchIndices,
		CONST(float*) stretchLengths,
		CONST(float*) invMasses,
		const uint numConstraints)
	{
		// �ж���Լ�����ʹ��������߳�
		GET_CUDA_ID(id, numConstraints);

		int idx1 = stretchIndices[2 * id];
		int idx2 = stretchIndices[2 * id + 1];
		// Ԥ�ھ���
		float expectedDistance = stretchLengths[id];

		glm::vec3 diff = predicted[idx1] - predicted[idx2];
		float distance = glm::length(diff);
		// ���������ĵ���
		float w1 = invMasses[idx1];
		float w2 = invMasses[idx2];
		// ������벻������������ && �������������������Ϊ0
		// �൱�����ĳ���Ǿ�ֹ�㣬���������������ر��Ϳ�����
		if (distance != expectedDistance && w1 + w2 > 0)
		{
			// ���漸�ж�ӦPBD���ĵĹ�ʽ(9)(10)
			glm::vec3 gradient = diff / (distance + EPSILON);
			// compliance is zero, therefore XPBD=PBD
			float denom = w1 + w2;
			float lambda = (distance - expectedDistance) / denom;
			glm::vec3 common = lambda * gradient;
			glm::vec3 correction1 = -w1 * common;
			glm::vec3 correction2 = w2 * common;


			int reorder = idx1 + idx2;
			// ���ǽ�correction1�ĸ����������ӵ�delats��
			AtomicAdd(deltas, idx1, correction1, reorder);
			AtomicAdd(deltas, idx2, correction2, reorder);
			/*
			�����д������ڼ���Լ��Ӧ�õĴ�����ÿ�μ���Լ��ʱdeltaCounts�����ж�Ӧ����ļ�����������1
			�ü������������ں�����Լ����������㷨����Ϊ  ����ݼ�������ֵ�����ɳ����ӻ��������
			*/
			atomicAdd(&deltaCounts[idx1], 1);
			atomicAdd(&deltaCounts[idx2], 1);
			//printf("correction[%d] = (%.2f,%.2f,%.2f)\n", idx1, correction1.x, correction1.y, correction1.z);
			//printf("correction[%d] = (%.2f,%.2f,%.2f)\n", idx2, correction2.x, correction2.y, correction2.z);
		}
	}

	void SolveStretch(
		glm::vec3* predicted,
		glm::vec3* deltas,
		int* deltaCounts,
		CONST(int*) stretchIndices, 
		CONST(float*) stretchLengths,
		CONST(float*) invMasses,
		const uint numConstraints)
	{
		ScopedTimerGPU timer("Solver_SolveStretch");
		CUDA_CALL(SolveStretch_Kernel, numConstraints)(predicted, deltas, deltaCounts, stretchIndices, stretchLengths, invMasses, numConstraints);
	}

	// ��������Լ��
	__global__ void SolveBending_Kernel(
		glm::vec3* predicted,
		glm::vec3* deltas,
		int* deltaCounts,
		CONST(uint*) bendingIndices,
		CONST(float*) bendingAngles,
		CONST(float*) invMass,
		const uint numConstraints,
		const float deltaTime)
	{
		// ����Լ����������ID
		GET_CUDA_ID(id, numConstraints);
		uint idx1 = bendingIndices[id * 4];
		uint idx2 = bendingIndices[id * 4+1];
		uint idx3 = bendingIndices[id * 4+2];
		uint idx4 = bendingIndices[id * 4+3];
		// ���������Ƕ�
		float expectedAngle = bendingAngles[id];
		// ��������
		float w1 = invMass[idx1];
		float w2 = invMass[idx2];
		float w3 = invMass[idx3];
		float w4 = invMass[idx4];

		glm::vec3 p1 = predicted[idx1];
		glm::vec3 p2 = predicted[idx2] - p1;
		glm::vec3 p3 = predicted[idx3] - p1;
		glm::vec3 p4 = predicted[idx4] - p1;
		glm::vec3 n1 = glm::normalize(glm::cross(p2, p3));
		glm::vec3 n2 = glm::normalize(glm::cross(p2, p4));

		float d = clamp(glm::dot(n1, n2), 0.0f, 1.0f);
		float angle = acos(d);
		// cross product for two equal vector produces NAN
		if (angle < EPSILON || isnan(d)) return;

		glm::vec3 q3 = (glm::cross(p2, n2) + glm::cross(n1, p2) * d) / (glm::length(glm::cross(p2, p3)) + EPSILON);
		glm::vec3 q4 = (glm::cross(p2, n1) + glm::cross(n2, p2) * d) / (glm::length(glm::cross(p2, p4)) + EPSILON);
		glm::vec3 q2 = -(glm::cross(p3, n2) + glm::cross(n1, p3) * d) / (glm::length(glm::cross(p2, p3)) + EPSILON)
			- (glm::cross(p4, n1) + glm::cross(n2, p4) * d) / (glm::length(glm::cross(p2, p4)) + EPSILON);
		glm::vec3 q1 = -q2 - q3 - q4;

		float xpbd_bend = d_params.bendCompliance / deltaTime / deltaTime;
		// ��ӦCSDN bending ��https://blog.csdn.net/weixin_43940314/article/details/129830991��
		float denom = xpbd_bend + (w1 * glm::dot(q1, q1) + w2 * glm::dot(q2, q2) + w3 * glm::dot(q3, q3) + w4 * glm::dot(q4, q4));
		if (denom < EPSILON) return; // ?
		float lambda = sqrt(1.0f - d * d) * (angle - expectedAngle) / denom;

		int reorder = idx1 + idx2 + idx3 + idx4;
		AtomicAdd(deltas, idx1, w1 * lambda * q1, reorder);
		AtomicAdd(deltas, idx2, w2 * lambda * q2, reorder);
		AtomicAdd(deltas, idx3, w3 * lambda * q3, reorder);
		AtomicAdd(deltas, idx4, w4 * lambda * q4, reorder);
		
		atomicAdd(&deltaCounts[idx1], 1);
		atomicAdd(&deltaCounts[idx2], 1);
		atomicAdd(&deltaCounts[idx3], 1);
		atomicAdd(&deltaCounts[idx4], 1);
	}

	void SolveBending(
		glm::vec3* predicted,
		glm::vec3* deltas,
		int* deltaCounts,
		CONST(uint*) bendingIndices,
		CONST(float*) bendingAngles,
		CONST(float*) invMass,
		const uint numConstraints,
		const float deltaTime)
	{
		ScopedTimerGPU timer("Solver_SolveBending");
		CUDA_CALL(SolveBending_Kernel, numConstraints)(predicted, deltas, deltaCounts, bendingIndices, bendingAngles, invMass, numConstraints, deltaTime);
	}

	// ������Լ��
	/*
	����Լ��������ģ��������ӡ���������������Ч��������ʵ�ָ�������µĸ��Ź�ϵ
	����̶������һ���֡�ά������֮������ӡ�����������α��
	*/
	__global__ void SolveAttachment_Kernel(
		glm::vec3* predicted,
		glm::vec3* deltas,
		int* deltaCounts,
		CONST(float*) invMass,
		CONST(int*) attachParticleIDs,
		CONST(int*) attachSlotIDs,
		CONST(glm::vec3*) attachSlotPositions,
		CONST(float*) attachDistances,
		const int numConstraints)
	{
		GET_CUDA_ID(id, numConstraints);

		uint pid = attachParticleIDs[id];
		// ��ȡ���Ų�λλ��
		glm::vec3 slotPos = attachSlotPositions[attachSlotIDs[id]];
		// d_params.longRangeStretchiness����������� ��������Ŀ������������
		float targetDist = attachDistances[id] * d_params.longRangeStretchiness;
		if (invMass[pid] == 0 && targetDist > 0) return;

		glm::vec3 pred = predicted[pid];
		glm::vec3 diff = pred - slotPos;
		float dist = glm::length(diff);
		// ��Ҫ����
		if (dist > targetDist)
		{
			//float coefficient = max(targetDist, dist - 0.1*d_params.particleDiameter);// 0.05 * targetDist + 0.95 * dist;
			glm::vec3 correction = -diff + diff / dist * targetDist;
			AtomicAdd(deltas, pid, correction, id);
			atomicAdd(&deltaCounts[pid], 1);
		}
	}

	void SolveAttachment(
		glm::vec3* predicted,
		glm::vec3* deltas,
		int* deltaCounts,
		CONST(float*) invMass,
		CONST(int*) attachParticleIDs,
		CONST(int*) attachSlotIDs,
		CONST(glm::vec3*) attachSlotPositions,
		CONST(float*) attachDistances,
		const int numConstraints)
	{
		ScopedTimerGPU timer("Solver_SolveAttach");
		CUDA_CALL(SolveAttachment_Kernel, numConstraints)(predicted, deltas, deltaCounts, 
			invMass, attachParticleIDs, attachSlotIDs, attachSlotPositions, attachDistances, numConstraints);
	}

	// Ӧ�������� deltas
	__global__ void ApplyDeltas_Kernel(glm::vec3* predicted, glm::vec3* deltas, int* deltaCounts)
	{
		GET_CUDA_ID(id, d_params.numParticles);

		float count = (float)deltaCounts[id];
		// Լ����������0
		if (count > 0)
		{
			// Ӧ�øı���
			predicted[id] += deltas[id] / count * d_params.relaxationFactor;
			// �ı�����Ϊ0
			deltas[id] = glm::vec3(0);
			// �������
			deltaCounts[id] = 0;
		}
	}

	void ApplyDeltas(glm::vec3* predicted, glm::vec3* deltas, int* deltaCounts)
	{
		ScopedTimerGPU timer("Solver_ApplyDeltas");
		CUDA_CALL(ApplyDeltas_Kernel, h_params.numParticles)(predicted, deltas, deltaCounts);
	}

	// ����Ħ��������������
	__device__ glm::vec3 ComputeFriction(glm::vec3 correction, glm::vec3 relVel,glm::vec3 friction1)
	{
		glm::vec3 friction = glm::vec3(0);
		// ��������������ģ
		float correctionLength = glm::length(correction);
		if (d_params.friction > 0 && correctionLength > 0)
		{
			// ��������������λ����
			glm::vec3 norm = correction / correctionLength;

			glm::vec3 tanVel = relVel - norm * glm::dot(relVel, norm);
			float tanLength = glm::length(tanVel);
			float maxTanLength = correctionLength * d_params.friction;

			friction = -tanVel * min(maxTanLength / tanLength, 1.0f)* friction1;
		}
		return friction;
	}

	__global__ void CollideSDF_Kernel(
		glm::vec3* predicted,			// Ԥ��λ��
		CONST(SDFCollider*) colliders,	// ��ײ��
		CONST(glm::vec3*) positions,	// λ��
		const uint numColliders,		// ��ײ������
		const float deltaTime)			// ʱ����
	{
		// �������������������߳�
		GET_CUDA_ID(id, d_params.numParticles);
		// ��ǰλ�ã����������ɣ�
		auto pos = positions[id];
		// Ԥ��λ��
		auto pred = predicted[id];
		for (int i = 0; i < numColliders; i++)
		{
			auto collider = colliders[i];
			// ������������
			glm::vec3 correction = collider.ComputeSDF(pred, d_params.collisionMargin);
			// Ӧ����������
			pred += correction;

			if (glm::dot(correction, correction) > 0)
			{
				//����Ħ����
				glm::vec3 friction1 = glm::vec3(5.0);
				// ��������ٶ�  Ԥ��λ�� - ��ǰλ�� - ��ײ֮���ٶ� * ʱ����
				glm::vec3 relVel = pred - pos - collider.VelocityAt(pred) * deltaTime;
				auto friction = ComputeFriction(correction, relVel,friction1);
				pred += friction;
			}
		}
		predicted[id] = pred;
	}

	void CollideSDF(
		glm::vec3* predicted,
		CONST(SDFCollider*) colliders,
		CONST(glm::vec3*) positions,
		const uint numColliders,
		const float deltaTime)
	{
		ScopedTimerGPU timer("Solver_CollideSDFs");
		if (numColliders == 0) return;
		
		CUDA_CALL(CollideSDF_Kernel, h_params.numParticles)(predicted, colliders, positions, numColliders, deltaTime);
	}
	
	// ��ײ����������
	__global__ void CollideParticles_Kernel(
		glm::vec3* deltas,
		int* deltaCounts,
		CONST(glm::vec3*) predicted,
		CONST(float*) invMasses,
		CONST(uint*) neighbors,
		CONST(glm::vec3*) positions)
	{
		// �����������������߳�
		GET_CUDA_ID(id, d_params.numParticles);

		glm::vec3 positionDelta = glm::vec3(0);
		int deltaCount = 0;
		glm::vec3 pred_i = predicted[id];
		glm::vec3 vel_i = (pred_i - positions[id]);
		float w_i = invMasses[id];

		for (int neighbor = id; neighbor < d_params.numParticles * d_params.maxNumNeighbors; neighbor += d_params.numParticles)
		{
			uint j = neighbors[neighbor];
			if (j > d_params.numParticles) break;

			float w_j = invMasses[j];
			float denom = w_i + w_j;
			if (denom <= 0) continue;

			glm::vec3 pred_j = predicted[j];
			glm::vec3 diff = pred_i - pred_j;
			float distance = glm::length(diff);
			if (distance >= d_params.particleDiameter) continue;

			glm::vec3 gradient = diff / (distance + EPSILON);
			float lambda = (distance - d_params.particleDiameter) / denom;
			glm::vec3 common = lambda * gradient;

			deltaCount++;
			positionDelta -= w_i * common;

			glm::vec3 relativeVelocity = vel_i - (pred_j - positions[j]);
			//
			glm::vec3 friction1 = glm::vec3(5.0);
			glm::vec3 friction = ComputeFriction(common, relativeVelocity, friction1);
			positionDelta += w_i * friction;
		}

		deltas[id] = positionDelta;
		deltaCounts[id] = deltaCount;
	}

	void CollideParticles(
		glm::vec3* deltas,
		int* deltaCounts,
		glm::vec3* predicted,
		CONST(float*) invMasses,
		CONST(uint*) neighbors,
		CONST(glm::vec3*) positions)
	{
		ScopedTimerGPU timer("Solver_CollideParticles");
		// �ȼ�����ײ
		CUDA_CALL(CollideParticles_Kernel, h_params.numParticles)(deltas, deltaCounts, predicted, invMasses, neighbors, positions);
		// Ӧ������
		CUDA_CALL(ApplyDeltas_Kernel, h_params.numParticles)(predicted, deltas, deltaCounts);
	}

	__global__ void Finalize_Kernel(
		glm::vec3* velocities,
		glm::vec3* positions,
		CONST(glm::vec3*) predicted,
		const float deltaTime)
	{
		// ���������������߳�ID
		GET_CUDA_ID(id, d_params.numParticles);
		// ��λ��
		glm::vec3 new_pos = predicted[id];
		// ��λ�ò����ٶ�
		glm::vec3 raw_vel = (new_pos - positions[id]) / deltaTime;
		// �ٶȵ�ģ
		float raw_vel_len = glm::length(raw_vel);
		// �����������ٶȣ���ʹ������ٶȼ���
		if (raw_vel_len > d_params.maxSpeed)
		{
			raw_vel = raw_vel / raw_vel_len * d_params.maxSpeed;
			new_pos = positions[id] + raw_vel * deltaTime;
			//printf("Limit vel[%.3f>%.3f] for id[%d]. Pred[%.3f,%.3f,%.3f], Pos[%.3f,%.3f,%.3f]\n", raw_vel_len, d_params.maxSpeed, id);
			//printf("new_pos %f %f %f\n", new_pos.x, new_pos.y, new_pos.z);
		}
		velocities[id] = raw_vel * (1 - d_params.damping * deltaTime);
		positions[id] = new_pos;
		//printf("new_pos %f %f %f\n", new_pos.x, new_pos.y, new_pos.z);
	}

	// ģ������첽  ִ�иú���(�൱�ڰ����е������������꣬����Ӧ�õ�λ����)
	void Finalize(
		glm::vec3* velocities, 
		glm::vec3* positions,
		CONST(glm::vec3*) predicted,
		const float deltaTime)
	{
		ScopedTimerGPU timer("Solver_Finalize");
		CUDA_CALL(Finalize_Kernel, h_params.numParticles)(velocities, positions, predicted, deltaTime);
	}

	// ���������η���
	__global__ void ComputeTriangleNormals(
		glm::vec3* normals,
		CONST(glm::vec3*) positions,
		CONST(uint*) indices,
		uint numTriangles)
	{
		// �������������������߳���
		GET_CUDA_ID(id, numTriangles);
		uint idx1 = indices[id * 3];
		uint idx2 = indices[id * 3+1];
		uint idx3 = indices[id * 3+2];

		auto p1 = positions[idx1];
		auto p2 = positions[idx2];
		auto p3 = positions[idx3];

		auto normal = glm::cross(p2 - p1, p3 - p1);
		//if (isnan(normal.x) || isnan(normal.y) || isnan(normal.z)) normal = glm::vec3(0, 1, 0);

		int reorder = idx1 + idx2 + idx3;
		AtomicAdd(normals, idx1, normal, reorder);
		AtomicAdd(normals, idx2, normal, reorder);
		AtomicAdd(normals, idx3, normal, reorder);
	}

	// ���㶥�㷨�ߣ���ʵ���ǹ�һ����
	__global__ void ComputeVertexNormals(glm::vec3* normals)
	{
		// �����������������߳���
		GET_CUDA_ID(id, d_params.numParticles);

		auto normal = glm::normalize(normals[id]);
		//normals[id] = glm::vec3(0,1,0);
		normals[id] = normal;
	}

	// ���㷨��
	void ComputeNormal(
		glm::vec3* normals,
		CONST(glm::vec3*) positions, 
		CONST(uint*) indices, 
		const uint numTriangles)
	{
		ScopedTimerGPU timer("Solver_UpdateNormals");
		if (h_params.numParticles)
		{
			// ��ͬ�������ڴ�
			cudaMemsetAsync(normals, 0, h_params.numParticles * sizeof(glm::vec3));
			// ��һ�������������η���
			CUDA_CALL(ComputeTriangleNormals, numTriangles)(normals, positions, indices, numTriangles);
			// �ڶ�������һ������
			CUDA_CALL(ComputeVertexNormals, h_params.numParticles)(normals);
		}
	}

}