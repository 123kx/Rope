#pragma once

#include <iostream>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "Component.hpp"
#include "Transform.hpp"

namespace Velvet
{
	using namespace std;

	class Actor
	{
	public:
		Actor();

		Actor(string _name);

		void Initialize(glm::vec3 position, glm::vec3 scale = glm::vec3(1), glm::vec3 rotation = glm::vec3(0),float mass=0.0f);

		void Start();

		void Update();

		void FixedUpdate();

		void OnDestroy();

		void AddComponent(shared_ptr<Component> component);

		void AddComponents(const initializer_list<shared_ptr<Component>>& newComponents);

		template <typename T>
		enable_if_t<is_base_of<Component, T>::value, T*> GetComponent()
		{
			T* result = nullptr;
			for (auto c : components)
			{
				result = dynamic_cast<T*>(c.get());
				if (result)
					return result;
			}
			return result;
		}

		template <typename T>
		enable_if_t<is_base_of<Component, T>::value, vector<T*>> GetComponents()
		{
			vector<T*> result;
			for (auto c : components)
			{
				auto item = dynamic_cast<T*>(c.get());
				if (item)
				{
					result.push_back(item);
				}
			}
			return result;
		}

	public:
		std::shared_ptr<Transform> transform = make_shared<Transform>(Transform(this));
		vector<shared_ptr<Component>> components;
		string name;
	};

}