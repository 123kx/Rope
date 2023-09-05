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
˼·���൱�ڰ���Ļ�����λ��������ά������
*/

namespace Velvet
{
	// ����
	struct Ray
	{
		// Դ��
		glm::vec3 origin;
		// ����
		glm::vec3 direction;
	};

	struct RaycastCollision
	{
		// �Ƿ���ײ
		bool collide = false;
		// ���������
		int objectIndex;
		// ����Դ��ľ���
		float distanceToOrigin;

	};

	class MouseGrabber
	{
	public:
		// ��ʼ��
		void Initialize(VtMergedBuffer<glm::vec3>* positions, VtBuffer<glm::vec3>* velocities, VtBuffer<float>* invMass)
		{
			m_positions = positions;
			m_velocities = velocities;
			m_invMass = invMass;
		}
		// ������꽻��
		void HandleMouseInteraction()
		{
			// �Ƿ���Ҫץȡ����
			bool shouldPickObject = Global::input->GetMouseDown(GLFW_MOUSE_BUTTON_LEFT);
			if (shouldPickObject)
			{
				Ray ray = GetMouseRay();
				// Ѱ������ĵ���Ϣ
				m_rayCollision = FindClosestVertexToRay(ray);
				// ���ȷ�����
				if (m_rayCollision.collide)
				{
					// �Ѿ���ץȡ
					m_isGrabbing = true;
					// ��ץȡ�������
					m_grabbedVertexMass = (*m_invMass)[m_rayCollision.objectIndex];
					// ��ץȡ���������Ϊ0
					(*m_invMass)[m_rayCollision.objectIndex] = 60;//��ǰ����Ϊ0
				}
			}
			// �ͷ�����
			bool shouldReleaseObject = Global::input->GetMouseUp(GLFW_MOUSE_BUTTON_LEFT);
			if (shouldReleaseObject && m_isGrabbing)
			{
				m_isGrabbing = false;
				(*m_invMass)[m_rayCollision.objectIndex] = m_grabbedVertexMass;
			}
		}

		// ����ץȡ��   ԭ
		//void UpdateGrappedVertex()
		//{
		//	if (m_isGrabbing)
		//	{
		//		Ray ray = GetMouseRay();
		//		// ��������������е�λ��			
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
				glm::vec3 target = Helper::Lerp(mousePos, curPos, 0.5f); // ��С��ֵȨ��

				// �������λ���ҵ����ڶ�������
				int startIndex = id - 1;//
				int endIndex = id + 1;
				if (startIndex < 0) startIndex = 0;
				if (endIndex >= m_positions->size()) endIndex = m_positions->size() - 1;

				for (int i = startIndex; i <= endIndex; i++)
				{
					auto position = (*m_positions)[i];
					(*m_positions)[i] = Helper::Lerp(target, position, 0.5f); // ��С��ֵȨ��
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

		// Ѱ�Ҿ�����������Ķ�����Ϣ
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