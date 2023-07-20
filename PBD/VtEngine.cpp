#include "VtEngine.hpp"

#include <algorithm>

#include "Scene.hpp"
#include "GUI.hpp"
#include "GameInstance.hpp"
#include "Input.hpp"

using namespace Velvet;

// ���GLFW����
void PrintGlfwError(int error, const char* description)
{
	std::cout << "Error(Glfw): Code({" << error << "}),{" << description << "}\n";
	//fmt::print("Error(Glfw): Code({}), {}\n", error, description);
}

VtEngine::VtEngine()
{
	Global::engine = this;
	// ���� glfw
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// �ز��� �����
	glfwWindowHint(GLFW_SAMPLES, 4);

	m_window = glfwCreateWindow(Global::Config::screenWidth, Global::Config::screenHeight, "Velvet", NULL, NULL);

	if (m_window == NULL)
	{
		std::cout << "Failed to create GLFW window\n";
		glfwTerminate();
		return;
	}

	glfwMakeContextCurrent(m_window); 
	glfwSwapInterval(0);
	// ���ڴ�С���Ļص�����
	glfwSetFramebufferSizeCallback(m_window, [](GLFWwindow* m_window, int width, int height) {
		glViewport(0, 0, width, height);
		});

	// �������ɼ���
	//glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// ���λ�ûص�����
	glfwSetCursorPosCallback(m_window, [](GLFWwindow* m_window, double xpos, double ypos) {
		Global::game->ProcessMouse(m_window, xpos, ypos);
		});

	// 
	glfwSetScrollCallback(m_window, [](GLFWwindow* m_window, double xoffset, double yoffset) {
		Global::game->ProcessScroll(m_window, xoffset, yoffset);
		});
	glfwSetErrorCallback(PrintGlfwError);

	// setup opengl
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD\n";
		return;
	}
	glViewport(0, 0, Global::Config::screenWidth, Global::Config::screenHeight);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	// ��ת��ͼ
	stbi_set_flip_vertically_on_load(true);

	// setup members
	m_gui = make_shared<GUI>(m_window);
	m_input = make_shared<Input>(m_window);
}

VtEngine::~VtEngine()
{
	m_gui->ShutDown();
	glfwTerminate();
}

void VtEngine::SetScenes(const vector<shared_ptr<Scene>>& initializers)
{
	scenes = initializers;
}

int VtEngine::Run()
{
	do 
	{
		std::cout <<
			"��{0:\-^{2}}��\n" <<
			"��{1: ^{2}}��\n" <<
			"��{0:\-^{2}}��\n" << "" << "Hello, Velvet!" << std::endl;
		// ����һ��GameInstanceʵ��
		m_game = make_shared<GameInstance>(m_window, m_gui);
		sceneIndex = m_nextSceneIndex;
		// ���ó�����Ϣ
		scenes[sceneIndex]->PopulateActors(m_game.get());
		scenes[sceneIndex]->onEnter.Invoke();
		// ��ѭ��
		m_game->Run();
		scenes[sceneIndex]->onExit.Invoke();
		scenes[sceneIndex]->ClearCallbacks();

		Resource::ClearCache();
		m_gui->ClearCallback();
	} while (m_game->pendingReset);

	return 0;
}

void VtEngine::Reset()
{
	m_game->pendingReset = true;
}

void VtEngine::SwitchScene(unsigned int _sceneIndex)
{
	m_nextSceneIndex = std::clamp(_sceneIndex, 0u, (unsigned int)scenes.size()-1);
	m_game->pendingReset = true;
}

glm::ivec2 VtEngine::windowSize()
{
	glm::ivec2 result;
	glfwGetWindowSize(m_window, &result.x, &result.y);
	return result;
}
