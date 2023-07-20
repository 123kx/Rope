#include <iostream>

#include "VtEngine.hpp"
#include "GameInstance.hpp"
#include "Resource.hpp"
#include "Scene.hpp"
#include "Helper.hpp"

using namespace Velvet;

class SceneClothSDF : public Scene
{
public:
    SceneClothSDF() { name = "Cloth / SDF Collision"; }

    void PopulateActors(GameInstance* game)  override
    {
        SpawnCameraAndLight(game);

        SpawnInfinitePlane(game);

        auto sphere = SpawnSphere(game);
        float radius = 0.0f;
        float mass = 2.0;
        sphere->Initialize(glm::vec3(0, radius, -1), glm::vec3(radius),glm::vec3(0),mass);
     /*   game->animationUpdate.Register([sphere, game, radius]() {
            float time = Timer::fixedDeltaTime() * Timer::physicsFrameCount();
            sphere->transform->position = glm::vec3(0, radius, -cos(time * 2));
            });*/
        sphere->transform->position = glm::vec3(0,1,3);
        int clothResolution = 12;
        auto cloth = SpawnCloth(game, clothResolution, 2);
        cloth->Initialize(glm::vec3(0, 2.5f, 0), glm::vec3(1.0));
#ifdef SOLVER_CPU
        auto clothObj = cloth->GetComponent<VtClothObject>();
#else
        auto clothObj = cloth->GetComponent<VtClothObjectGPU>();
#endif
        if (clothObj) clothObj->SetAttachedIndices({ 0, clothResolution-3 });
    }
};

int main() 
{	
    //=====================================
        // 1. Create graphics
        //=====================================
    auto engine = make_shared<VtEngine>();

    //=====================================
    // 2. Instantiate actors
    //=====================================

    vector<shared_ptr<Scene>> scenes = {
        make_shared<SceneClothSDF>(),
        //make_shared<ScenePremitiveRendering>(),
    };
    engine->SetScenes(scenes);

    //=====================================
    // 3. Run graphics
    //=====================================
    return engine->Run();
}