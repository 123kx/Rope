#pragma once

#include <iostream>
#include <unordered_map>
#include <string>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <glm/glm.hpp>

#include "External/stb_image.h"
#include "Mesh.hpp"
#include "Material.hpp"

namespace Velvet
{
	class Resource
	{
	public:
		// 加载贴图
		static unsigned int LoadTexture(const string& path)
		{
			std::cout << path << std::endl;
			if (textureCache.count(path) > 0)
			{
				return textureCache[path];
			}

            unsigned int textureID;
            glGenTextures(1, &textureID);
			textureCache[path] = textureID;

            int width, height, nrComponents;
            unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrComponents, 0);
			if (data == nullptr)
			{
				data = stbi_load((defaultTexturePath+path).c_str(), &width, &height, &nrComponents, 0);
			}
            if (data)
            {
				GLenum internalFormat = GL_RED;
				GLenum dataFormat = GL_RED;

				if (nrComponents == 3)
				{
					internalFormat = GL_SRGB;
					dataFormat = GL_RGB;
				}
				else if (nrComponents == 4)
				{
					internalFormat = GL_SRGB_ALPHA;
					dataFormat = GL_RGBA;
				}

				glBindTexture(GL_TEXTURE_2D, textureID);
				glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, dataFormat, GL_UNSIGNED_BYTE, data);
                glGenerateMipmap(GL_TEXTURE_2D);

				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

                stbi_image_free(data);
            }
            else
            {
				//fmt::print("Error(Resource): Texture failed to load at path({})\n", path);
				std::cout << "Error(Resource): Texture failed to load at path({" << path << "})\n";
                stbi_image_free(data);
            }

            return textureID;
		}
		// 加载网格
		static shared_ptr<Mesh> LoadMesh(const std::string& path)
		{
			if (meshCache.count(path) > 0)
			{
				return meshCache[path];
			}

			vector<glm::vec3> vertices;
			vector<glm::vec3> normals;
			vector<glm::vec2> texCoords;
			vector<unsigned int> indices;

			Assimp::Importer importer;
			const aiScene* scene = importer.ReadFile(defaultMeshPath + path, aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
			// check for errors
			if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
			{
				scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);

				if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
				{	
					std::cout << "Error(Resource) Fail to load mesh ({" << path << "})\n";
					return shared_ptr<Mesh>();
				}
			}
			aiMesh* mesh = scene->mMeshes[0];

			int drawCount = mesh->mNumVertices;

			// 提取每个三角形的顶点信息
			for (unsigned int i = 0; i < mesh->mNumVertices; i++)
			{
				// 位置
				vertices.push_back(AdaptVector(mesh->mVertices[i]));
				// 法线
				if (mesh->HasNormals())
				{
					normals.push_back(AdaptVector(mesh->mNormals[i]));
				}
				else
				{
					std::cout << "Normals not found\n";
					exit(-1);
				}
				// UV
				if (mesh->mTextureCoords[0]) // does the mesh contain texture coordinates?
				{
					texCoords.push_back(AdaptVector(mesh->mTextureCoords[0][i]));
				}
				else
				{
					texCoords.push_back(glm::vec2(0.0f, 0.0f));
				}
			}
			
			// 遍历每个三角形面，获取顶点索引
			for (unsigned int i = 0; i < mesh->mNumFaces; i++)
			{
				aiFace face = mesh->mFaces[i];
				// 检索面部的所有索引并将其存储在索引向量中
				for (unsigned int j = 0; j < face.mNumIndices; j++)
					indices.push_back(face.mIndices[j]);
			}

			auto result = shared_ptr<Mesh>(new Mesh(vertices, normals, texCoords, indices));
			meshCache[path] = result;
			return result;
		}
		// 加载材质
		static shared_ptr<Material> LoadMaterial(const string& path, bool includeGeometryShader = false)
		{
			if (matCache.count(path))
			{
				return matCache[path];
			}
			string vertexCode = LoadText(defaultMaterialPath + path + ".vert");
			if (vertexCode.length() == 0) vertexCode = LoadText(path + ".vert");
			if (vertexCode.length() == 0)
			{
				std::cout << "Error(Resource): material.vertex not found ({" << path << "})\n";
				exit(-1);
			}

			string fragmentCode = LoadText(defaultMaterialPath + path + ".frag");
			if (fragmentCode.length() == 0) fragmentCode = LoadText(path + ".frag");
			if (fragmentCode.length() == 0)
			{
				std::cout << "Error(Resource): material.fragment not found ({" << path << "})\n";
				exit(-1);
			}

			string geometryCode;
			if (includeGeometryShader)
			{
				geometryCode = LoadText(defaultMaterialPath + path + ".geom");
				if (geometryCode.length() == 0) geometryCode = LoadText(path + ".geom");
				if (geometryCode.length() == 0)
				{
					std::cout << "Error(Resource): material.geometry not found({" << path << "})\n";
					exit(-1);
				}
			}
			auto result = make_shared<Material>(vertexCode, fragmentCode, geometryCode);
			matCache[path] = result;
			result->name = path;
			return result;
		}
		// 加载文本
		static string LoadText(const string& path)
		{
			// 1. retrieve the vertex/fragment source code from filePath
			std::string code;
			std::ifstream file;
			// ensure ifstream objects can throw exceptions:
			file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
			try
			{
				// open files
				file.open(path);
				std::stringstream vShaderStream;
				// read file's buffer contents into streams
				vShaderStream << file.rdbuf();
				// close file handlers
				file.close();
				// convert stream into string
				code = vShaderStream.str();
			}
			catch (std::ifstream::failure& e)
			{
				e;
			}
			return code;
		}

		static void ClearCache()
		{
			// only mat needs to be clear
			matCache.clear();
			meshCache.clear();
		}

	private:
		static inline glm::vec3 AdaptVector(const aiVector3D& input)
		{
			return glm::vec3(input.x, input.y, input.z);
		}

		static inline glm::vec2 AdaptVector(const aiVector2D& input)
		{
			return glm::vec2(input.x, input.y);
		}
		// 贴图备份
		static inline unordered_map<string, unsigned int> textureCache;
		// 网格备份存储
		static inline unordered_map<string, shared_ptr<Mesh>> meshCache;
		// 材质备份
		static inline unordered_map<string, shared_ptr<Material>> matCache;

		static inline string defaultTexturePath = "Assets/Texture/";
		static inline string defaultMeshPath = "Assets/Model/";
		static inline string defaultMaterialPath = "Assets/Shader/";
	};
}