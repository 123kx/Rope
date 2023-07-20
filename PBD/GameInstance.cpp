#include "GameInstance.hpp"

#include <functional>
#include <iomanip>
#include "Helper.hpp"
#include "Camera.hpp"
#include "Input.hpp"
#include "RenderPipeline.hpp"
#include "GUI.hpp"
#include "Timer.hpp"
#include "VtEngine.hpp"
#include "Resource.hpp"

using namespace Velvet;

// 构造函数
GameInstance::GameInstance(GLFWwindow* window, shared_ptr<GUI> gui)
{
	Global::game = this;

	// setup members
	m_window = window;
	m_gui = gui;
	m_renderPipeline = make_shared<RenderPipeline>();
	m_timer = make_shared<Timer>();

	Timer::StartTimer("GAME_INSTANCE_INIT");
}

shared_ptr<Actor> GameInstance::AddActor(shared_ptr<Actor> actor)
{
	m_actors.push_back(actor);
	return actor;
}

shared_ptr<Actor> GameInstance::CreateActor(const string& name)
{
	auto actor = shared_ptr<Actor>(new Actor(name));
	return AddActor(actor);
}

int GameInstance::Run()
{
	// Print actors
	if (0)
	{
		std::cout << "Total actors: {" << m_actors.size() << "}\n";
		for (auto actor : m_actors)
		{
			std::cout << " + {" << actor->name  << "}\n";
			for (auto component : actor->components)
			{
				std::cout << " |-- {" << component->name << "}\n";
			}
		}
		std::cout << std::endl;
	}

	Initialize();
	MainLoop();
	Finalize();

	return 0;
}

unsigned int GameInstance::depthFrameBuffer()
{
	return m_renderPipeline->depthTex;
}

// 窗口大小
glm::ivec2 Velvet::GameInstance::windowSize()
{
	glm::ivec2 result;
	glfwGetWindowSize(m_window, &result.x, &result.y);
	return result;
}

bool Velvet::GameInstance::windowMinimized()
{
	auto size = windowSize();
	return (size.x < 1 || size.y < 1);
}

void GameInstance::ProcessMouse(GLFWwindow* m_window, double xpos, double ypos)
{
	onMouseMove.Invoke(xpos, ypos);
}

void GameInstance::ProcessScroll(GLFWwindow* m_window, double xoffset, double yoffset)
{
	onMouseScroll.Invoke(xoffset, yoffset);
}

void GameInstance::ProcessKeyboard(GLFWwindow* m_window)
{
	Global::input->ToggleOnKeyDown(GLFW_KEY_H, Global::gameState.hideGUI);

	if (Global::input->GetKey(GLFW_KEY_ESCAPE))
	{
		glfwSetWindowShouldClose(m_window, true);
	}
	if (Global::input->GetKeyDown(GLFW_KEY_O))
	{
		Global::gameState.step = !Global::gameState.step;
		Global::gameState.pause = !Global::gameState.pause;
	}
	for (int i = 0; i < 9; i++)
	{
		if (Global::input->GetKeyDown(GLFW_KEY_1 + i))
		{
			Global::engine->SwitchScene(i);
		}
	}
	if (Global::input->GetKeyDown(GLFW_KEY_R))
	{
		Global::engine->Reset();
	}
}

// 初始化，所有Actor均处于start状态
void GameInstance::Initialize()
{
	// 所有物体都开始
	for (const auto& go : m_actors)
	{
		// Actor的start调用当前actor的所有组件的start
		go->Start();
	}
}

void GameInstance::MainLoop()
{
	double initTime = Timer::EndTimer("GAME_INSTANCE_INIT") * 1000;
	std::cout << "Info(GameInstance): Initialization success within " << std::fixed << std::setprecision(2) << initTime << " ms. Enter main loop." << std::endl;
	// 渲染循环
	while (!glfwWindowShouldClose(m_window) && !pendingReset)
	{
		if (windowMinimized())
		{
			glfwPollEvents();
			continue;
		}
		// 输入事件
		ProcessKeyboard(m_window);

		// 初始化
		glClearColor(skyColor.x, skyColor.y, skyColor.z, skyColor.w);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glPolygonMode(GL_FRONT_AND_BACK, Global::gameState.renderWireframe ? GL_LINE : GL_FILL);

		Timer::StartTimer("CPU_TIME");
		Timer::UpdateDeltaTime();

		// GUI循环
		if (!Global::gameState.hideGUI) 
			m_gui->OnUpdate();

		// 场景循环
		if (!Global::gameState.pause)
		{
			Timer::NextFrame();
			
			if (Timer::NextFixedFrame())
			{
				for (const auto& go : m_actors) go->FixedUpdate();

				animationUpdate.Invoke();

				if (Global::gameState.step)
				{
					Global::gameState.pause = true;
					Global::gameState.step = false;
				}
			}

			for (const auto& go : m_actors) go->Update();
		}

		Global::input->OnUpdate();

		godUpdate.Invoke();

		Timer::EndTimer("CPU_TIME");

		// 渲染流水线的渲染
		m_renderPipeline->Render();
		if (!Global::gameState.hideGUI) m_gui->Render();

		// Check and call events and swap the buffers
		glfwSwapBuffers(m_window);
		glfwPollEvents();
	}
}

// 终止
void GameInstance::Finalize()
{
	for (const auto& go : m_actors)
	{
		go->OnDestroy();
	}
}

