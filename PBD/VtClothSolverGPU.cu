#include "VtClothSolverGPU.cuh"
#include "Common.cuh"
#include "Common.hpp"
#include "Timer.hpp"

// 调试使用，如调试完，切记*****注释*****


using namespace std;

namespace Velvet
{
	// constant 指定常量内存，它在GPU上有特殊的访问方式，可以提供更快的读取性能
	// 但是，这些参数在整个CUDA程序的执行过程中，都是不可修改的，但可以被读取
	__device__ __constant__ VtSimParams d_params;
	VtSimParams h_params;

	__device__ inline void AtomicAdd(glm::vec3* address, int index, glm::vec3 val, int reorder)
	{
		// reorder保证了原子操作的正确性
		/*
		可以进行验证，使用了reorder之后，r1、r2、r3的值唯一（可以随便举例子验算）
		*/
		int r1 = reorder % 3;
		int r2 = (reorder + 1) % 3;
		int r3 = (reorder + 2) % 3;
		// 实现了修正值的分量级别的累加操作
		atomicAdd(&(address[index].x) + r1, val[r1]);
		atomicAdd(&(address[index].x) + r2, val[r2]);
		atomicAdd(&(address[index].x) + r3, val[r3]);
	}

	// 设置模拟参数时间
	void SetSimulationParams(VtSimParams* hostParams)
	{
		ScopedTimerGPU timer("Solver_SetParams");
		// checkCudaErrors检查CUDA函数的返回值并处理任何错误
		// cudaMemcpyToSymbolAsync 用于将数据从主机内存异步复制到设备的常量内存
		/*
		将主机内存中的数据通过异步操作复制到设备的常量内存"d_params"中，
		这样设备上的函数可以使用常量内存中的数据进行计算，而不需要从主机内存读取数据，从而提高性能
		*/
		checkCudaErrors(cudaMemcpyToSymbolAsync(d_params, hostParams, sizeof(VtSimParams)));
		h_params = *hostParams;
	}

	// 初始化位置
	__global__ void InitializePositions_Kernel(glm::vec3* positions, const int start, const int count, const glm::mat4 modelMatrix)
	{
		GET_CUDA_ID(id, count);
		// 将位置转换到世界坐标下
		positions[start + id] = modelMatrix * glm::vec4(positions[start+id], 1);
	}

	void InitializePositions(glm::vec3* positions, const int start, const int count, const glm::mat4 modelMatrix)
	{
		ScopedTimerGPU timer("Solver_Initialize");
		CUDA_CALL(InitializePositions_Kernel, count)(positions, start, count, modelMatrix);
	}

	// 预测位置  其中输入的position是不能更改的，所以为CONST
	__global__ void PredictPositions_Kernel(
		glm::vec3* predicted,
		glm::vec3* velocities,
		CONST(glm::vec3*) positions,
		const float deltaTime)
	{
		/*
		有多少个粒子，就创建多少个线程，
		*/
		GET_CUDA_ID(id, d_params.numParticles);

		//glm::vec3 gravity = glm::vec3(0, -10, 0);
		// 计算速度
		velocities[id] += d_params.gravity * deltaTime;
		// 计算位移
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

	// 处理拉伸约束 ( deltas其实就是delta_p )
	__global__ void SolveStretch_Kernel(
		glm::vec3* predicted,
		glm::vec3* deltas,
		int* deltaCounts,
		CONST(int*) stretchIndices,
		CONST(float*) stretchLengths,
		CONST(float*) invMasses,
		const uint numConstraints)
	{
		// 有多少约束，就创建多少线程
		GET_CUDA_ID(id, numConstraints);

		int idx1 = stretchIndices[2 * id];
		int idx2 = stretchIndices[2 * id + 1];
		// 预期距离
		float expectedDistance = stretchLengths[id];

		glm::vec3 diff = predicted[idx1] - predicted[idx2];
		float distance = glm::length(diff);
		// 顶点质量的倒数
		float w1 = invMasses[idx1];
		float w2 = invMasses[idx2];
		// 如果距离不等于期望距离 && 两个顶点的质量倒数不为0
		// 相当于如果某点是静止点，将它的质量设置特别大就可以了
		if (distance != expectedDistance && w1 + w2 > 0)
		{
			// 下面几行对应PBD论文的公式(9)(10)
			glm::vec3 gradient = diff / (distance + EPSILON);
			// compliance is zero, therefore XPBD=PBD
			float denom = w1 + w2;
			float lambda = (distance - expectedDistance) / denom;
			glm::vec3 common = lambda * gradient;
			glm::vec3 correction1 = -w1 * common;
			glm::vec3 correction2 = w2 * common;


			int reorder = idx1 + idx2;
			// 就是将correction1的各个分量都加到delats上
			AtomicAdd(deltas, idx1, correction1, reorder);
			AtomicAdd(deltas, idx2, correction2, reorder);
			/*
			这两行代码用于计数约束应用的次数，每次计算约束时deltaCounts数组中对应顶点的计数器会增加1
			该计数器可以用于后续的约束求解或控制算法的行为  如根据计数器的值调整松弛因子或迭代次数
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

	// 处理弯曲约束
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
		// 根据约束数量计算ID
		GET_CUDA_ID(id, numConstraints);
		uint idx1 = bendingIndices[id * 4];
		uint idx2 = bendingIndices[id * 4+1];
		uint idx3 = bendingIndices[id * 4+2];
		uint idx4 = bendingIndices[id * 4+3];
		// 期望弯曲角度
		float expectedAngle = bendingAngles[id];
		// 质量倒数
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
		// 对应CSDN bending （https://blog.csdn.net/weixin_43940314/article/details/129830991）
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

	// 链接物约束
	/*
	附着约束常用来模拟刚体连接、绳索、弹簧物理效果，用于实现各种情况下的附着关系
	例如固定物体的一部分、维持物体之间的连接、控制物体的形变等
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
		// 获取附着槽位位置
		glm::vec3 slotPos = attachSlotPositions[attachSlotIDs[id]];
		// d_params.longRangeStretchiness长程拉伸参数 利用它对目标距离进行缩放
		float targetDist = attachDistances[id] * d_params.longRangeStretchiness;
		if (invMass[pid] == 0 && targetDist > 0) return;

		glm::vec3 pred = predicted[pid];
		glm::vec3 diff = pred - slotPos;
		float dist = glm::length(diff);
		// 需要修正
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

	// 应用修正量 deltas
	__global__ void ApplyDeltas_Kernel(glm::vec3* predicted, glm::vec3* deltas, int* deltaCounts)
	{
		GET_CUDA_ID(id, d_params.numParticles);

		float count = (float)deltaCounts[id];
		// 约束次数大于0
		if (count > 0)
		{
			// 应用改变量
			predicted[id] += deltas[id] / count * d_params.relaxationFactor;
			// 改变量置为0
			deltas[id] = glm::vec3(0);
			// 次数清空
			deltaCounts[id] = 0;
		}
	}

	void ApplyDeltas(glm::vec3* predicted, glm::vec3* deltas, int* deltaCounts)
	{
		ScopedTimerGPU timer("Solver_ApplyDeltas");
		CUDA_CALL(ApplyDeltas_Kernel, h_params.numParticles)(predicted, deltas, deltaCounts);
	}

	// 计算摩擦力的修正向量
	__device__ glm::vec3 ComputeFriction(glm::vec3 correction, glm::vec3 relVel,glm::vec3 friction1)
	{
		glm::vec3 friction = glm::vec3(0);
		// 计算修正向量的模
		float correctionLength = glm::length(correction);
		if (d_params.friction > 0 && correctionLength > 0)
		{
			// 计算修正向量单位向量
			glm::vec3 norm = correction / correctionLength;

			glm::vec3 tanVel = relVel - norm * glm::dot(relVel, norm);
			float tanLength = glm::length(tanVel);
			float maxTanLength = correctionLength * d_params.friction;

			friction = -tanVel * min(maxTanLength / tanLength, 1.0f)* friction1;
		}
		return friction;
	}

	__global__ void CollideSDF_Kernel(
		glm::vec3* predicted,			// 预测位置
		CONST(SDFCollider*) colliders,	// 碰撞体
		CONST(glm::vec3*) positions,	// 位置
		const uint numColliders,		// 碰撞体数量
		const float deltaTime)			// 时间间隔
	{
		// 根据粒子数量，创建线程
		GET_CUDA_ID(id, d_params.numParticles);
		// 当前位置？？？（存疑）
		auto pos = positions[id];
		// 预测位置
		auto pred = predicted[id];
		for (int i = 0; i < numColliders; i++)
		{
			auto collider = colliders[i];
			// 计算修正向量
			glm::vec3 correction = collider.ComputeSDF(pred, d_params.collisionMargin);
			// 应用修正向量
			pred += correction;

			if (glm::dot(correction, correction) > 0)
			{
				//设置摩擦力
				glm::vec3 friction1 = glm::vec3(5.0);
				// 计算相对速度  预测位置 - 当前位置 - 碰撞之后速度 * 时间间隔
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
	
	// 碰撞产生的修正
	__global__ void CollideParticles_Kernel(
		glm::vec3* deltas,
		int* deltaCounts,
		CONST(glm::vec3*) predicted,
		CONST(float*) invMasses,
		CONST(uint*) neighbors,
		CONST(glm::vec3*) positions)
	{
		// 根据粒子数量计算线程
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
		// 先计算碰撞
		CUDA_CALL(CollideParticles_Kernel, h_params.numParticles)(deltas, deltaCounts, predicted, invMasses, neighbors, positions);
		// 应用修正
		CUDA_CALL(ApplyDeltas_Kernel, h_params.numParticles)(predicted, deltas, deltaCounts);
	}

	__global__ void Finalize_Kernel(
		glm::vec3* velocities,
		glm::vec3* positions,
		CONST(glm::vec3*) predicted,
		const float deltaTime)
	{
		// 根据粒子数计算线程ID
		GET_CUDA_ID(id, d_params.numParticles);
		// 新位置
		glm::vec3 new_pos = predicted[id];
		// 新位置产生速度
		glm::vec3 raw_vel = (new_pos - positions[id]) / deltaTime;
		// 速度的模
		float raw_vel_len = glm::length(raw_vel);
		// 如果超过最大速度，就使用最大速度即可
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

	// 模拟最后异步  执行该函数(相当于把所有的修正都计算完，最终应用到位置上)
	void Finalize(
		glm::vec3* velocities, 
		glm::vec3* positions,
		CONST(glm::vec3*) predicted,
		const float deltaTime)
	{
		ScopedTimerGPU timer("Solver_Finalize");
		CUDA_CALL(Finalize_Kernel, h_params.numParticles)(velocities, positions, predicted, deltaTime);
	}

	// 计算三角形法线
	__global__ void ComputeTriangleNormals(
		glm::vec3* normals,
		CONST(glm::vec3*) positions,
		CONST(uint*) indices,
		uint numTriangles)
	{
		// 根据三角形数量计算线程数
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

	// 计算顶点法线（其实就是归一化）
	__global__ void ComputeVertexNormals(glm::vec3* normals)
	{
		// 根据粒子数量计算线程数
		GET_CUDA_ID(id, d_params.numParticles);

		auto normal = glm::normalize(normals[id]);
		//normals[id] = glm::vec3(0,1,0);
		normals[id] = normal;
	}

	// 计算法线
	void ComputeNormal(
		glm::vec3* normals,
		CONST(glm::vec3*) positions, 
		CONST(uint*) indices, 
		const uint numTriangles)
	{
		ScopedTimerGPU timer("Solver_UpdateNormals");
		if (h_params.numParticles)
		{
			// 先同步法线内存
			cudaMemsetAsync(normals, 0, h_params.numParticles * sizeof(glm::vec3));
			// 第一步：计算三角形法线
			CUDA_CALL(ComputeTriangleNormals, numTriangles)(normals, positions, indices, numTriangles);
			// 第二步：归一化法线
			CUDA_CALL(ComputeVertexNormals, h_params.numParticles)(normals);
		}
	}

}