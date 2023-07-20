#pragma once

#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Component.hpp"
#include "Global.hpp"
#include "Actor.hpp"

using namespace std;

namespace Velvet
{
	class Camera : public Component
	{
	public:
		float zoom = 45.0f;

		Camera()
		{
			name = __func__;
			Global::camera = this;
		}
		// λ��
		glm::vec3 position() const
		{
			return actor->transform->position;
		}
		// ǰ
		glm::vec3 front() const
		{
			const glm::vec3 kFront = glm::vec3(0.0f, 0.0f, -1.0f);
			return Helper::RotateWithDegree(kFront, actor->transform->rotation);
		}
		// ��
		glm::vec3 up() const
		{
			const glm::vec3 kUp = glm::vec3(0.0f, 1.0f, 0.0f);
			return Helper::RotateWithDegree(kUp, actor->transform->rotation);
		}
		// �ӿ�
		glm::mat4 view() const
		{
			auto trans = actor->transform;
			auto rotation = trans->rotation;
			auto result = glm::lookAt(position(), position() + front(), up());
			return result;
		}
		// ͶӰ
		glm::mat4 projection() const
		{
			auto size = Global::game->windowSize();
			auto screenAspect = (float)size.x / (float)size.y;
			return glm::perspective(glm::radians(zoom), screenAspect, 0.01f,
				100.0f);
		}
	};
}