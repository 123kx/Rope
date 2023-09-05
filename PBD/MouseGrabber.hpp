#pragma once

#include "glm/glm.hpp"

#include "Component.hpp"
#include "Helper.hpp"
#include "VtBuffer.hpp"
#include "Global.hpp"
#include "Input.hpp"
#include "Actor.hpp"
#include "Timer.hpp"
#include "VtEngine.hpp"
#include "Camera.hpp"

/*
思路：相当于把屏幕坐标的位置逆向到三维物体上
*/

namespace Velvet
{
	// 射线
	struct Ray
	{
		// 源点
		glm::vec3 origin;
		// 方向
		glm::vec3 direction;
	};

	struct RaycastCollision
	{
		// 是否碰撞
		bool collide = false;
		// 物体的索引
		int objectIndex;
		// 距离源点的距离
		float distanceToOrigin;

	};

	class MouseGrabber
	{
	public:
		// 初始化
		void Initialize(VtMergedBuffer<glm::vec3>* positions, VtBuffer<glm::vec3>* velocities, VtBuffer<float>* invMass)
		{
			m_positions = positions;
			m_velocities = velocities;
			m_invMass = invMass;
		}
		// 处理鼠标交互
		void HandleMouseInteraction()
		{
			// 是否需要抓取物体
			bool shouldPickObject = Global::input->GetMouseDown(GLFW_MOUSE_BUTTON_LEFT);
			if (shouldPickObject)
			{
				Ray ray = GetMouseRay();
				// 寻找最近的点信息
				m_rayCollision = FindClosestVertexToRay(ray);
				// 鼠标确定点击
				if (m_rayCollision.collide)
				{
					// 已经被抓取
					m_isGrabbing = true;
					// 被抓取点的质量
					m_grabbedVertexMass = (*m_invMass)[m_rayCollision.objectIndex];
					// 将抓取点的质量置为0
					(*m_invMass)[m_rayCollision.objectIndex] = 60;//以前质量为0
				}
			}
			// 释放物体
			bool shouldReleaseObject = Global::input->GetMouseUp(GLFW_MOUSE_BUTTON_LEFT);
			if (shouldReleaseObject && m_isGrabbing)
			{
				m_isGrabbing = false;
				(*m_invMass)[m_rayCollision.objectIndex] = m_grabbedVertexMass;
			}
		}

		// 更新抓取点   原
		//void UpdateGrappedVertex()
		//{
		//	if (m_isGrabbing)
		//	{
		//		Ray ray = GetMouseRay();
		//		// 鼠标在世界坐标中的位置			
		//		glm::vec3 mousePos = ray.origin + ray.direction * m_rayCollision.distanceToOrigin;
		//		int id = m_rayCollision.objectIndex;
		//		auto curPos = (*m_positions)[id];
		//		glm::vec3 target = Helper::Lerp(mousePos, curPos, 0.8f);

		//		(*m_positions)[id] = target;
		//		(*m_velocities)[id] = (target - curPos) / Timer::fixedDeltaTime();
		//	}
		//}

		void UpdateGrappedVertex()
		{
			if (m_isGrabbing)
			{
				Ray ray = GetMouseRay();
				glm::vec3 mousePos = ray.origin + ray.direction * m_rayCollision.distanceToOrigin;
				int id = m_rayCollision.objectIndex;
				auto curPos = (*m_positions)[id];
				glm::vec3 target = Helper::Lerp(mousePos, curPos, 0.5f); // 调小插值权重

				// 根据鼠标位置找到相邻顶点索引
				int startIndex = id - 1;//
				int endIndex = id + 1;
				if (startIndex < 0) startIndex = 0;
				if (endIndex >= m_positions->size()) endIndex = m_positions->size() - 1;

				for (int i = startIndex; i <= endIndex; i++)
				{
					auto position = (*m_positions)[i];
					(*m_positions)[i] = Helper::Lerp(target, position, 0.5f); // 调小插值权重
					(*m_velocities)[i] = ((*m_positions)[i] - position) / Timer::fixedDeltaTime();
				}
			}
		}

	private:
		bool m_isGrabbing = false;
		float m_grabbedVertexMass = 0;
		RaycastCollision m_rayCollision;

		VtMergedBuffer<glm::vec3>* m_positions;
		VtBuffer<glm::vec3>* m_velocities;
		VtBuffer<float>* m_invMass;

		// 寻找距离射线最近的顶点信息
		RaycastCollision FindClosestVertexToRay(Ray ray)
		{
			int result = -1;
			float minDistanceToView = FLT_MAX;

			for (int i = 0; i < m_positions->size(); i++)
			{
				const auto& position = (*m_positions)[i];
				float distanceToView = glm::dot(ray.direction, position - ray.origin);
				float distanceToRay = glm::length(glm::cross(ray.direction, position - ray.origin));

				if (distanceToRay < Global::simParams.particleDiameter && distanceToView < minDistanceToView)
				{
					result = i;
					minDistanceToView = distanceToView;
				}
			}
			return RaycastCollision{ result >= 0, result, minDistanceToView };
		}



		Ray GetMouseRay()
		{
			glm::vec2 screenPos = Global::input->GetMousePos();
			// [0, 1]
			auto windowSize = Global::engine->windowSize();
			auto normalizedScreenPos = 2.0f * screenPos / glm::vec2(windowSize.x, windowSize.y) - 1.0f;
			normalizedScreenPos.y = -normalizedScreenPos.y;

			glm::mat4 invVP = glm::inverse(Global::camera->projection() * Global::camera->view());
			glm::vec4 nearPointRaw = invVP * glm::vec4(normalizedScreenPos, 0, 1);
			glm::vec4 farPointRaw = invVP * glm::vec4(normalizedScreenPos, 1, 1);

			glm::vec3 nearPoint = glm::vec3(nearPointRaw.x, nearPointRaw.y, nearPointRaw.z) / nearPointRaw.w;
			glm::vec3 farPoint = glm::vec3(farPointRaw.x, farPointRaw.y, farPointRaw.z) / farPointRaw.w;
			glm::vec3 direction = glm::normalize(farPoint - nearPoint);

			return Ray{ nearPoint, direction };
		}
	};
}