#pragma once

#include <string>
#include <functional>

#include "GameInstance.hpp"
#include "Input.hpp"
#include "Resource.hpp"
#include "Actor.hpp"
#include "PlayerController.hpp"
#include "MeshRenderer.hpp"
#include "MaterialProperty.hpp"
#include "Collider.hpp"
#include "VtClothObjectCPU.hpp"
#include "VtClothObjectGPU.hpp"
#include "ParticleInstancedRenderer.hpp"
#include "ParticleGeometryRenderer.hpp"

#include<cmath>


//#define SOLVER_CPU

namespace Velvet
{
	class Scene
	{
	public:
		std::string name = "BaseScene";

		virtual void PopulateActors(GameInstance* game) = 0;

		void ClearCallbacks() { onEnter.Clear(); onExit.Clear(); }

		VtCallback<void()> onEnter;
		VtCallback<void()> onExit;

	protected:

		template <class T>
		void ModifyParameter(T* ptr, T value)
		{
			onEnter.Register([this, ptr, value]() {
				T prev = *ptr;
				*ptr = value;
				onExit.Register([ptr, prev, value]() {
					//fmt::print("Revert ptr[{}] from {} to value {}\n", (int)ptr, value, prev);
					*ptr = prev;
					});
				});
		}

		// 创建摄像机和灯光
		void SpawnCameraAndLight(GameInstance* game)
		{
			// Camera
			auto camera = SpawnCamera(game);
			camera->Initialize(glm::vec3(0.35, 23.3,57.2),
				glm::vec3(1),
				glm::vec3(-21, 12.25, 0));

			// Light
			auto light = SpawnLight(game);
			//position,scale,rotation
			light->Initialize(glm::vec3(8.5f,50.0f, 20.5f),
				glm::vec3(6.2f),
				glm::vec3(20, 30, 0));
			auto lightComp = light->GetComponent<Light>();

			SpawnDebug(game);

			//game->postUpdate.push_back([light, lightComp, game]() {
			//	//light->transform->position = glm::vec3(sin(glfwGetTime()), 4.0, cos(glfwGetTime()));
			//	light->transform->rotation = glm::vec3(10 * sin(Timer::elapsedTime()) - 10, 0, 0);
			//	light->transform->position = glm::vec3(2.5 * sin(Timer::elapsedTime()), 4.0, 2.5 * cos(Timer::elapsedTime()));
			//	if (Global::input->GetKeyDown(GLFW_KEY_UP))
			//	{
			//		fmt::print("Outer: {}\n", lightComp->outerCutoff++);
			//	}
			//	if (Global::input->GetKeyDown(GLFW_KEY_DOWN))
			//	{
			//		fmt::print("Outer: {}\n", lightComp->outerCutoff--);
			//	}
			//	if (Global::input->GetKeyDown(GLFW_KEY_RIGHT))
			//	{
			//		fmt::print("Inner: {}\n", lightComp->innerCutoff++);
			//	}
			//	if (Global::input->GetKeyDown(GLFW_KEY_LEFT))
			//	{
			//		fmt::print("Inner: {}\n", lightComp->innerCutoff--);
			//	}
			//	});
		}

		// 为了调试阴影使用的调试FBO
		void SpawnDebug(GameInstance* game)
		{
			auto quad = game->CreateActor("Debug Quad");
			{
				auto debugMat = Resource::LoadMaterial("_ShadowDebug");
				{
					float near_plane = 1.0f, far_plane = 7.5f;
					debugMat->SetFloat("near_plane", near_plane);
					debugMat->SetFloat("far_plane", far_plane);
					debugMat->SetTexture("depthMap", game->depthFrameBuffer());
				}
				vector<float> quadVertices = {
					// positions        // texture Coords
					-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
					-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
					 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,

					-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
					 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
					 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
				};
				vector<unsigned int> attributes = { 3,2 };
				auto quadMesh = make_shared<Mesh>(attributes, quadVertices);
				shared_ptr<MeshRenderer> renderer(new MeshRenderer(quadMesh, debugMat));
				quad->AddComponent(renderer);
				renderer->enabled = false;

				game->godUpdate.Register([renderer]() {
					if (Global::input->GetKeyDown(GLFW_KEY_X))
					{
						//fmt::print("Info(Scene): Visualize shadow texutre. Turn on/off by key X.\n");
						std::cout << "Info(Scene): Visualize shadow texutre. Turn on/off by key X.\n";
						renderer->enabled = !renderer->enabled;
					}
					});
			}
		}
		// 创建布料网格
		shared_ptr<Mesh> GenerateClothMesh(int resolution)
		{
			vector<glm::vec3> vertices;
			vector<glm::vec3> normals;
			vector<glm::vec2> uvs;
			vector<unsigned int> indices;

			const float clothSize = 2.0f;

			for (int y = 0; y <= resolution; y++)
			{
				for (int x = 0; x <= resolution; x++)
				{
					vertices.push_back(clothSize * glm::vec3((float)x / (float)resolution - 0.5f, -(float)y / (float)resolution, 0));
					normals.push_back(glm::vec3(0, 0, 1));
					uvs.push_back(glm::vec2((float)x / (float)resolution, (float)y / (float)resolution));
				}
			}

			auto VertexIndexAt = [resolution](int x, int y) {
				return x * (resolution + 1) + y;
			};

			for (int x = 0; x < resolution; x++)
			{
				for (int y = 0; y < resolution; y++)
				{
					indices.push_back(VertexIndexAt(x, y));
					indices.push_back(VertexIndexAt(x + 1, y));
					indices.push_back(VertexIndexAt(x, y + 1));

					indices.push_back(VertexIndexAt(x, y + 1));
					indices.push_back(VertexIndexAt(x + 1, y));
					indices.push_back(VertexIndexAt(x + 1, y + 1));
				}
			}
			auto mesh = make_shared<Mesh>(vertices, normals, uvs, indices);
			//std::cout << "创建网格数：" << indices.size() << std::endl;
			return mesh;
		}

		/*这个方法就是绳子很长时就会出现错误
//	shared_ptr<Mesh> GenerateRopeMesh(int vertLineNum)*/
//{
//vector<glm::vec3> vertices;
//vector<glm::vec3> normals;
//vector<glm::vec2> uvs;
//vector<unsigned int> indices;
//const glm::vec3& bornPos = glm::vec3(0.0f);
//int length = 10;
//float radius = 0.3f;
//// int vertLineNum = length +1;
//vertices.resize(vertLineNum * 6);
//normals.resize(vertLineNum * 6);
//uvs.resize(vertLineNum * 6);
//indices.resize(length * 6 * 2 * 3);
//float M_PI = std::acos(-1.0f);
//
//		int triangleIterator = 0;
//		for (int i = 0; i < vertLineNum; i++)
//		{
//			for (int j = 0; j < 6; j++)
//			{
//
//				int vertIndex = i * 6 + j;
//				vertices[vertIndex] = glm::vec3(bornPos.x, bornPos.y, bornPos.z + i);
//				double angle = M_PI / 3 * j;
//				vertices[vertIndex].x += radius * cos(angle);
//				vertices[vertIndex].y += radius * sin(angle);
//				uvs.push_back(glm::vec2(vertLineNum * 6));
//					normals.push_back(glm::vec3(0, 0, 1));
//					uvs[vertIndex] = glm::vec2(1.0f/6*j,1.0f/length*i);
//
//				if (i < vertLineNum - 1)
//				{
//					indices[triangleIterator++] = vertIndex;
//					indices[triangleIterator++] = j == 5 ? vertIndex + 1 : vertIndex + 7;
//					indices[triangleIterator++] = j == 5 ? vertIndex - 5 : vertIndex + 1;
//					indices[triangleIterator++] = vertIndex;
//					indices[triangleIterator++] = vertIndex + 6;
//					indices[triangleIterator++] = j == 5 ? vertIndex + 1 : vertIndex + 7;
//				}
//			}
//		}
//		auto mesh = make_shared<Mesh>(vertices, normals, uvs, indices);
//		return mesh;
//
//	}
	//	创建绳子模型
	//	 一段直接生成网格

		shared_ptr<Mesh> GenerateRopeMesh(int vertLineNum)
		{
			vector<glm::vec3> vertices;
			vector<glm::vec3> normals;
			vector<glm::vec2> uvs;
			vector<unsigned int> indices;
			const glm::vec3 bornPos = glm::vec3(0.0f);
			int length = 100;
			float radius = 0.3f;
			vertices.reserve(vertLineNum * 6);
			normals.reserve(vertLineNum * 6);
			uvs.reserve(vertLineNum * 6);
			indices.reserve(length * 6 * 2 * 3);
			float M_PI = std::acos(-1.0f);

			for (int i = 0; i < vertLineNum; i++)
			{
				for (int j = 0; j < 6; j++)
				{
					int vertIndex = i * 6 + j;
					float angle = M_PI / 3 * j;
					glm::vec3 vertex(bornPos.x, bornPos.y, bornPos.z + i);
					vertex.x += radius * cos(angle);
					vertex.y += radius * sin(angle);

					vertices.push_back(vertex);
					normals.push_back(glm::vec3(0, 0, 1));
					uvs.push_back(glm::vec2(1.0f / 6 * j, 1.0f / length * i));

					if (i < vertLineNum - 1)
					{
						// 添加六面体的连接索引
						int nextRowIndex = i + 1;
						int nextRowVertIndex = nextRowIndex * 6 + j;

						indices.push_back(vertIndex);
						indices.push_back(nextRowVertIndex);
						indices.push_back(j == 5 ? nextRowIndex * 6 : nextRowVertIndex + 1);

						indices.push_back(vertIndex);
						indices.push_back(j == 5 ? nextRowIndex * 6 : nextRowVertIndex + 1);
						indices.push_back(j == 5 ? i * 6 : vertIndex + 1);
					}
				}
			}

			auto mesh = make_shared<Mesh>(vertices, normals, uvs, indices);
			return mesh;
		}			
		//圆柱
		//float M_PI = 3.14;
		//
		//shared_ptr<Mesh> GenerateRopeMesh(int vertLineNum)
		//{
		//	vector<glm::vec3> vertices;
		//	vector<glm::vec3> normals;
		//	vector<glm::vec2> uvs;
		//	vector<unsigned int> indices;
		//	const glm::vec3& bornPos = glm::vec3(0.0f);
		//	int length = 10;
		//	float radius = 0.3f;
		//	vertices.reserve(length * vertLineNum);
		//	normals.reserve(length * vertLineNum);
		//	uvs.reserve(length * vertLineNum);
		//	indices.reserve((length - 1) * 2 * 3 * vertLineNum);

		//	float angleStep = 2 * M_PI / vertLineNum;

		//	for (int i = 0; i < length; i++)
		//	{
		//		float posY = bornPos.y + i;
		//		for (int j = 0; j < vertLineNum; j++)
		//		{
		//			float posX = bornPos.x + radius * cos(j * angleStep);
		//			float posZ = bornPos.z + radius * sin(j * angleStep);
		//			vertices.push_back(glm::vec3(posX, posY, posZ));
		//			normals.push_back(glm::vec3(0, 0, 1));
		//			uvs.push_back(glm::vec2(j / static_cast<float>(vertLineNum - 1), i / static_cast<float>(length - 1)));

		//			if (i < length - 1)
		//			{
		//				// Create indices for triangle strips
		//				indices.push_back(i * vertLineNum + j);
		//				indices.push_back(i * vertLineNum + (j + 1) % vertLineNum);
		//				indices.push_back((i + 1) * vertLineNum + j);

		//				indices.push_back(i * vertLineNum + (j + 1) % vertLineNum);
		//				indices.push_back((i + 1) * vertLineNum + (j + 1) % vertLineNum);
		//				indices.push_back((i + 1) * vertLineNum + j);
		//			}
		//		}
		//	}

		//	auto mesh = make_shared<Mesh>(vertices, normals, uvs, indices);
		//	return mesh;
		//}

		//	分段生成网格
		/*shared_ptr<Mesh> GenerateRopeMesh(int vertLineNum, int segmentCount)
		{
			vector<glm::vec3> vertices;
			vector<glm::vec3> normals;
			vector<glm::vec2> uvs;
			vector<unsigned int> indices;
			const glm::vec3& bornPos = glm::vec3(0.0f);
			int length = 10;
			float radius = 0.3f;
			int totalVertices = vertLineNum * 6 * segmentCount;
			int totalIndices = length * 6 * 2 * 3 * segmentCount;
			vertices.reserve(totalVertices);
			normals.reserve(totalVertices);
			uvs.reserve(totalVertices);
			indices.reserve(totalIndices);
			float M_PI = 3.1415926f;
			int triangleIterator = 0;
			for (int s = 0; s < segmentCount; s++)
			{
				for (int i = 0; i < vertLineNum; i++)
				{
					for (int j = 0; j < 6; j++)
					{
						int vertIndex = s * vertLineNum * 6 + i * 6 + j;
						vertices.emplace_back(bornPos.x, bornPos.y, bornPos.z + s * length + i);
						double angle = M_PI / 3 * j;
						vertices[vertIndex].x += radius * cos(angle);
						vertices[vertIndex].y += radius * sin(angle);
						uvs.emplace_back(1.0f / 6 * j, 1.0f / length * (s * length + i));

						if (i < vertLineNum - 1)
						{
							indices.emplace_back(vertIndex);
							indices.emplace_back(j == 5 ? vertIndex + 1 : vertIndex + 7);
							indices.emplace_back(j == 5 ? vertIndex - 5 : vertIndex + 1);
							indices.emplace_back(vertIndex);
							indices.emplace_back(vertIndex + 6);
							indices.emplace_back(j == 5 ? vertIndex + 1 : vertIndex + 7);
							triangleIterator += 6;
						}
					}
				}
			}
			auto mesh = make_shared<Mesh>(vertices, normals, uvs, indices);
			return mesh;
		}*/


		/*	shared_ptr<Mesh> GenerateRopeMesh(int vertLineNum, int segmentCount)
			{
				vector<glm::vec3> vertices;
				vector<glm::vec3> normals;
				vector<glm::vec2> uvs;
				vector<unsigned int> indices;

				const glm::vec3& bornPos = glm::vec3(0.0f);
				int length = 10;
				float radius = 0.3f;
				int vertLineNumPerSegment = vertLineNum / segmentCount;
				vertices.resize(vertLineNum * 6);
				normals.resize(vertLineNum * 6);
				uvs.resize(vertLineNum * 6);
				indices.resize(length * 6 * 2 * 3 * segmentCount);
				float M_PI = 3.1415926f;
				int triangleIterator = 0;

				for (int seg = 0; seg < segmentCount; seg++)
				{
					int startIndex = seg * vertLineNumPerSegment * 6;
					int nextStartIndex = (seg + 1) * vertLineNumPerSegment * 6;

					for (int i = 0; i < vertLineNumPerSegment; i++)
					{
						for (int j = 0; j < 6; j++)
						{
							int vertIndex = startIndex + i * 6 + j;
							vertices[vertIndex] = glm::vec3(bornPos.x, bornPos.y, bornPos.z + seg * vertLineNumPerSegment + i);
							double angle = M_PI / 3 * j;
							vertices[vertIndex].x += radius * cos(angle);
							vertices[vertIndex].y += radius * sin(angle);
							uvs[vertIndex] = glm::vec2(1.0f / 6 * j, 1.0f / length * (seg * vertLineNumPerSegment + i));

							if (i < vertLineNumPerSegment - 1)
							{
								indices[triangleIterator++] = vertIndex;
								indices[triangleIterator++] = j == 5 ? vertIndex + 1 : vertIndex + 7;
								indices[triangleIterator++] = j == 5 ? vertIndex - 5 : vertIndex + 1;
								indices[triangleIterator++] = vertIndex;
								indices[triangleIterator++] = vertIndex + 6;
								indices[triangleIterator++] = j == 5 ? vertIndex + 1 : vertIndex + 7;
							}

							// 连接段
							if (seg < segmentCount - 1 && i == vertLineNumPerSegment - 1 && j < 5)
							{
								int nextVertIndex = nextStartIndex + j;
								indices[triangleIterator++] = vertIndex;
								indices[triangleIterator++] = vertIndex + 1;
								indices[triangleIterator++] = nextVertIndex + 1;
								indices[triangleIterator++] = vertIndex;
								indices[triangleIterator++] = nextVertIndex + 1;
								indices[triangleIterator++] = nextVertIndex;
							}
						}
					}
				}
				auto mesh = make_shared<Mesh>(vertices, normals, uvs, indices);
				return mesh;
			}*/
			// 创建不规则网格
		shared_ptr<Mesh> GenerateClothMeshIrregular(int resolution)
		{
			vector<glm::vec3> vertices;
			vector<glm::vec3> normals;
			vector<glm::vec2> uvs;
			vector<unsigned int> indices;
			const float clothSize = 2.0f;
			float noiseSize = 1.0f / resolution * 0.4f;

			auto IsBoundary = [resolution](int x, int y) {
				return x == 0 || y == 0 || x == resolution || y == resolution;
			};

			auto VertexIndexAt = [resolution](int x, int y) {
				return x * (resolution + 1) + y;
			};

			auto Angle = [](glm::vec3 left, glm::vec3 mid, glm::vec3 right) {
				auto line1 = left - mid;
				auto line2 = right - mid;
				return acos(glm::dot(line1, line2));
			};

			for (int y = 0; y <= resolution; y++)
			{
				for (int x = 0; x <= resolution; x++)
				{
					glm::vec2 noise = IsBoundary(x, y) ? glm::vec2(0) : noiseSize * glm::vec2(Helper::Random(), Helper::Random());
					glm::vec2 uv = noise + glm::vec2((float)x / (float)resolution, (float)y / (float)resolution);
					auto vertex = glm::vec3(uv.x - 0.5f, -uv.y, 0);

					vertices.push_back(clothSize * (vertex));
					normals.push_back(glm::vec3(0, 0, 1));
					uvs.push_back(uv);
				}
			}

			for (int y = 0; y < resolution; y++)
			{
				for (int x = 0; x < resolution; x++)
				{
					if (x < resolution && y < resolution)
					{
						auto pos1 = vertices[VertexIndexAt(x, y)];
						auto pos2 = vertices[VertexIndexAt(x + 1, y)];
						auto pos3 = vertices[VertexIndexAt(x, y + 1)];
						auto pos4 = vertices[VertexIndexAt(x + 1, y + 1)];

						auto angle1 = Angle(pos3, pos1, pos2);
						auto angle2 = Angle(pos1, pos2, pos4);
						auto angle3 = Angle(pos1, pos3, pos4);
						auto angle4 = Angle(pos3, pos4, pos2);

						if (angle1 + angle4 > angle2 + angle3)
						{
							indices.push_back(VertexIndexAt(x, y));
							indices.push_back(VertexIndexAt(x + 1, y + 1));
							indices.push_back(VertexIndexAt(x, y + 1));

							indices.push_back(VertexIndexAt(x, y));
							indices.push_back(VertexIndexAt(x + 1, y));
							indices.push_back(VertexIndexAt(x + 1, y + 1));
						}
						else
						{
							indices.push_back(VertexIndexAt(x, y));
							indices.push_back(VertexIndexAt(x + 1, y));
							indices.push_back(VertexIndexAt(x, y + 1));

							indices.push_back(VertexIndexAt(x, y + 1));
							indices.push_back(VertexIndexAt(x + 1, y));
							indices.push_back(VertexIndexAt(x + 1, y + 1));
						}
					}
				}
			}
			auto mesh = make_shared<Mesh>(vertices, normals, uvs, indices);
			return mesh;
		}

		shared_ptr<Mesh> GenerateRopeMesh()
		{
			auto mesh = Resource::LoadMesh("quad.obj");
			return mesh;
		}

		// 创建布料
		shared_ptr<Actor> SpawnCloth(GameInstance* game, int resolution=100, int textureFile = 1, shared_ptr<VtClothSolverGPU> solver = nullptr)
		{
			auto cloth = game->CreateActor("Cloth Generated");

			auto material = Resource::LoadMaterial("_Default");
			material->Use();
			material->doubleSided = true;

			MaterialProperty materialProperty;

			//	std::string fileName = "fabric" + std::to_string(std::clamp(textureFile, 1, 3)) + ".jpg";

			std::string fileName = "mabu.png";
			//std::string fileName = "fabric1.jpg";
			auto texture = Resource::LoadTexture(fileName);

			materialProperty.preRendering = [texture](Material* mat) {
				mat->SetVec3("material.tint", glm::vec3(0.0f, 0.5f, 1.0f));
				mat->SetBool("material.useTexture", true);
				mat->SetTexture("material.diffuse", texture);
				mat->specular = 0.01f;
			};

			// 创建网格

			//auto mesh = GenerateClothMesh(resolution);
			//auto mesh = GenerateClothMeshIrregular(resolution);
			//auto mesh = GenerateRopeMesh();
			//绳子  
			auto mesh = GenerateRopeMesh(100);//创建网格

			auto renderer = make_shared<MeshRenderer>(mesh, material, true);//渲染
			renderer->SetMaterialProperty(materialProperty);//材质

			//auto prenderer = make_shared<ParticleRenderer>();
			auto prenderer = make_shared<ParticleGeometryRenderer>();

			// 添加布料解算器
#ifdef SOLVER_CPU
			auto clothObj = make_shared<VtClothObject>(resolution);
#else
			if (solver == nullptr)
			{
				solver = make_shared<VtClothSolverGPU>();
				cloth->AddComponent(solver);
			}
			auto clothObj = make_shared<VtClothObjectGPU>(resolution, solver);
#endif

			cloth->AddComponents({ renderer, clothObj, prenderer });

			return cloth;
		}

		// 创建球体
		shared_ptr<Actor> SpawnSphere(GameInstance* game)
		{
			auto sphere = game->CreateActor("Sphere");
			MaterialProperty materialProperty;
			materialProperty.preRendering = [](Material* mat) {
				mat->SetVec3("material.tint", glm::vec3(1.0));
				mat->SetBool("material.useTexture", false);
			};

			auto material = Resource::LoadMaterial("_Default");

			auto mesh = Resource::LoadMesh("sphere.obj");
			auto renderer = make_shared<MeshRenderer>(mesh, material, true);
			renderer->SetMaterialProperty(materialProperty);
			auto collider = make_shared<Collider>(ColliderType::Sphere);
			sphere->AddComponents({ renderer, collider });
			return sphere;
		}


		// 创建灯光
		shared_ptr<Actor> SpawnLight(GameInstance* game)
		{
			// 先创建一个Actor
			auto actor = game->CreateActor("Prefab Light");
			// 灯光的mesh
			auto mesh = Resource::LoadMesh("cylinder.obj");
			// 
			auto material = Resource::LoadMaterial("Assets/Shader/UnlitWhite");
			auto renderer = make_shared<MeshRenderer>(mesh, material);
			auto light = make_shared<Light>();

			actor->AddComponents({ renderer, light });
			return actor;
		}

		// 创建摄像机
		shared_ptr<Actor> SpawnCamera(GameInstance* game)
		{
			auto actor = game->CreateActor("Prefab Camera");
			auto camera = make_shared<Camera>();
			auto controller = make_shared<PlayerController>();
			actor->AddComponents({ camera, controller });
			return actor;
		}

		// 创建平面
		shared_ptr<Actor> SpawnInfinitePlane(GameInstance* game)
		{
			auto infPlane = game->CreateActor("Infinite Plane");

			auto mat = Resource::LoadMaterial("InfinitePlane");
			mat->noWireframe = true;
			// Plane: ax + by + cz + d = 0
			mat->SetVec4("_Plane", glm::vec4(0, 1, 0, 0));

			const vector<glm::vec3> vertices = {
				glm::vec3(1,1,0), glm::vec3(-1,-1,0), glm::vec3(-1,1,0), glm::vec3(1,-1,0) };
			const vector<unsigned int> indices = { 2,1,0, 3, 0, 1 };

			auto mesh = make_shared<Mesh>(vertices, vector<glm::vec3>(), vector<glm::vec2>(), indices);
			auto renderer = make_shared<MeshRenderer>(mesh, mat);
			auto collider = make_shared<Collider>(ColliderType::Plane);
			infPlane->AddComponents({ renderer, collider });
			return infPlane;
		}

		shared_ptr<Actor> SpawnColoredCube(GameInstance* game, glm::vec3 color = glm::vec3(1.0f))
		{
			auto cube = game->CreateActor("Cube");
			auto material = Resource::LoadMaterial("_Default");

			MaterialProperty materialProperty;
			materialProperty.preRendering = [color](Material* mat) {
				mat->SetVec3("material.tint", color);
				mat->SetBool("material.useTexture", false);
			};

			auto mesh = Resource::LoadMesh("cube.obj");
			auto renderer = make_shared<MeshRenderer>(mesh, material, true);
			renderer->SetMaterialProperty(materialProperty);

			auto collider = make_shared<Collider>(ColliderType::Cube);
			cube->AddComponents({ renderer, collider });
			return cube;
		}

		shared_ptr<Actor> SpawnTestRope(GameInstance* game, shared_ptr<Mesh> mesh, glm::vec3 color = glm::vec3(0.5f))
		{
			auto cube = game->CreateActor("Rope");
			auto material = Resource::LoadMaterial("_Default");

			MaterialProperty materialProperty;
			materialProperty.preRendering = [color](Material* mat) {
				mat->SetVec3("material.tint", color);
				mat->SetBool("material.useTexture", false);
			};

			// auto mesh = Resource::LoadMesh("cube.obj");
			auto renderer = make_shared<MeshRenderer>(mesh, material, true);
			renderer->SetMaterialProperty(materialProperty);

			auto collider = make_shared<Collider>(ColliderType::Cube);
			cube->AddComponents({ renderer, collider });
			return cube;
		}
	};
}