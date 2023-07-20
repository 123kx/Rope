#pragma once

#include <glm/glm.hpp>
#include <sstream>
#include <iomanip>

#include <glm\ext\matrix_transform.hpp>

namespace Velvet
{
	namespace Helper
	{
		// vec3 to string 
		inline std::string to_string(const glm::vec3& p) {
			std::stringstream ss;
			ss << "[" << std::fixed << std::setprecision(2) << p.x << ", "
				<< std::fixed << std::setprecision(2) << p.y << ", "
				<< std::fixed << std::setprecision(2) << p.z << "]";
			return ss.str();
		}

		// vec2 to string
		inline std::string to_string(const glm::vec2& p) {
			std::stringstream ss;
			ss << "[" << std::fixed << std::setprecision(2) << p.x << ", "
				<< std::fixed << std::setprecision(2) << p.y << "]";
			return ss.str();
		}

		// totate by mat
		inline glm::mat4 RotateWithDegree(glm::mat4 result, const glm::vec3& rotation) 
		{
			result = glm::rotate(result, glm::radians(rotation.y), glm::vec3(0, 1, 0));
			result = glm::rotate(result, glm::radians(rotation.z), glm::vec3(0, 0, 1));
			result = glm::rotate(result, glm::radians(rotation.x), glm::vec3(1, 0, 0));

			return result;
		}

		//
		inline glm::vec3 RotateWithDegree(glm::vec3 result, const glm::vec3& rotation)
		{
			glm::mat4 rotationMatrix(1);
			rotationMatrix = RotateWithDegree(rotationMatrix, rotation);
			result = rotationMatrix * glm::vec4(result, 0.0f);
			return glm::normalize(result);
		}

		inline float Random(float min = 0, float max = 1)
		{
			float zeroToOne = (float)rand() / RAND_MAX;
			return min + zeroToOne * (max - min);
		}
		// unit vector
		inline glm::vec3 RandomUnitVector()
		{
			const float pi = 3.1415926535;
			float phi = Random(0, pi * 2.0f);
			float theta = Random(0, pi * 2.0f);

			float cosTheta = cos(theta);
			float sinTheta = sin(theta);

			float cosPhi = cos(phi);
			float sinPhi = sin(phi);

			return glm::vec3(cosTheta * sinPhi, cosPhi, sinTheta * sinPhi);
		}

		template <class T>
		T Lerp(T value1, T value2, float a)
		{
			a = min(max(a, 0.0f), 1.0f);
			return a * value2 + (1 - a) * value1;
		}
	}
}