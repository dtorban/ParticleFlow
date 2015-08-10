/*
 * HurricaneApp.cpp
 *
 *  Created on: Aug 6, 2015
 *      Author: dtorban
 */

#include <HurricaneApp.h>
#include "vrbase/scenes/render/BasicRenderedScene.h"
#include "vrbase/scenes/MeshScene.h"
#include "PFVis/scenes/ParticleScene.h"

using namespace vrbase;
using namespace PFVis::partflow;
using namespace std;

HurricaneApp::HurricaneApp() : PartFlowApp () {
	AppBase::init();

	vector<glm::vec3> vertices;
	vertices.push_back(glm::vec3(-1.0f, 0.0, -1.0));
	vertices.push_back(glm::vec3(0.0f, 1.0, -1.0));
	vertices.push_back(glm::vec3(0.0f, 0.0, 0.0));

	vertices.push_back(glm::vec3(0.0f, 0.0, 0.0));
	vertices.push_back(glm::vec3(0.0f, 1.0, -1.0));
	vertices.push_back(glm::vec3(1.0f, 0.0, -1.0));

	vertices.push_back(glm::vec3(0.0f, 0.0, 0.0));
	vertices.push_back(glm::vec3(0.0f, -1.0, -1.0));
	vertices.push_back(glm::vec3(-1.0f, 0.0, -1.0));

	vertices.push_back(glm::vec3(0.0f, 0.0, 0.0));
	vertices.push_back(glm::vec3(1.0f, 0.0, -1.0));
	vertices.push_back(glm::vec3(0.0f, -1.0, -1.0));

	vector<unsigned int> indices;
	for (int f = 0; f < vertices.size(); f++)
	{
		indices.push_back(f);
	}

	_mesh = MeshRef(new Mesh(vertices, indices));
}

HurricaneApp::~HurricaneApp() {
}

SceneRef HurricaneApp::createAppScene(int threadId, MinVR::WindowRef window)
{
	MeshScene* mesh = new MeshScene(_mesh);
	SceneRef scene = SceneRef(mesh);
	scene = SceneRef(new ParticleScene(scene, mesh));
	scene = SceneRef(new BasicRenderedScene(scene));
	return scene;
}
