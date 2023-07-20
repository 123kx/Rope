#pragma once

#include <iostream>
#include <string>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

using namespace std;
/*
它主要分为四个部分：场景、GUI、物体实例、输入
*/

namespace Velvet
{
	class Scene;
	class GUI;
	class GameInstance;
	class Input;

	class VtEngine
	{
	public:
		VtEngine();
		~VtEngine();

		// 运行
		int Run();
		// 重置
		void Reset();
		// 切换场景
		void SwitchScene(unsigned int sceneIndex);
		// 设置场景
		void SetScenes(const vector<shared_ptr<Scene>>& scenes);
		// 窗口大小
		glm::ivec2 windowSize();
		// 场景数组
		vector<shared_ptr<Scene>> scenes;
		// 场景索引
		unsigned int sceneIndex = 0;
	private:
		unsigned int m_nextSceneIndex = 0;
		GLFWwindow* m_window = nullptr;
		shared_ptr<GUI> m_gui;
		shared_ptr<GameInstance> m_game;
		shared_ptr<Input> m_input;
	};
}