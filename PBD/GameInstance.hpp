#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <functional>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "Component.hpp"
#include "Common.hpp"


namespace Velvet
{
	using namespace std;

	class Light;
	class RenderPipeline;
	class GUI;
	class Actor;
	class Timer;

	class GameInstance
	{
	public:
		GameInstance(GLFWwindow* window, shared_ptr<GUI> gui);
		GameInstance(const GameInstance&) = delete;

		shared_ptr<Actor> AddActor(shared_ptr<Actor> actor);
		shared_ptr<Actor> CreateActor(const string& name);

		int Run();

		void ProcessMouse(GLFWwindow* m_window, double xpos, double ypos);
		void ProcessScroll(GLFWwindow* m_window, double xoffset, double yoffset);
		void ProcessKeyboard(GLFWwindow* m_window);

		// 获取所有组件的指针
		template <typename T>
		enable_if_t<is_base_of<Component, T>::value, vector<T*>> FindComponents()
		{
			vector<T*> result;
			for (auto actor : m_actors)
			{
				auto component = actor->GetComponents<T>();
				if (component.size() > 0)
				{
					result.insert(result.end(), component.begin(), component.end());
				}
			}
			return result;
		}

	public:
		unsigned int depthFrameBuffer();
		glm::ivec2 windowSize();
		bool windowMinimized();

		VtCallback<void(double, double)> onMouseScroll;
		VtCallback<void(double, double)> onMouseMove;
		VtCallback<void()> animationUpdate;
		VtCallback<void()> godUpdate; // update when main logic is paused (for debugging purpose)
		VtCallback<void()> onFinalize;

		bool pendingReset = false;
		glm::vec4 skyColor = glm::vec4(0.4,0.6,0.8,1.0);

	private:
		// 初始化
		void Initialize();
		// 主循环
		void MainLoop();
		// 结束
		void Finalize();

	private:
		// 窗口
		GLFWwindow* m_window = nullptr;
		// GUI
		shared_ptr<GUI> m_gui;
		// Timer
		shared_ptr<Timer> m_timer;
		// 存储所有actor指针
		vector<shared_ptr<Actor>> m_actors;
		// 渲染管线
		shared_ptr<RenderPipeline> m_renderPipeline;



	};
}