/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <viewer/include/ViewerApp.h>
#include <vector>

#include "ViewerAppScene.h"
#include "vrbase/scenes/render/BasicRenderedScene.h"
#include "vrbase/scenes/MeshScene.h"
#include "vrbase/scenes/BlankScene.h"

ViewerApp::ViewerApp() : PFVis::partflow::PartFlowApp() {

	AppBase::init();

	std::vector<glm::vec3> vertices;
		vertices.push_back(glm::vec3(-1.0f, -1.0, 0.0));
		vertices.push_back(glm::vec3(-1.0f, 1.0, 0.0));
		vertices.push_back(glm::vec3(1.0f, 1.0, 0.0));

		vertices.push_back(glm::vec3(1.0f, 1.0, 0.0));
		vertices.push_back(glm::vec3(1.0f, -1.0, 0.0));
		vertices.push_back(glm::vec3(-1.0f, -1.0, 0.0));

		std::vector<unsigned int> indices;
		for (int f = 0; f < vertices.size(); f++)
		{
			//vertices[f] += glm::vec3(pos, 0.0f);
			indices.push_back(f);
		}

		vrbase::MeshRef	mesh = vrbase::MeshRef(new vrbase::Mesh(vertices, indices));
		_mesh = mesh;
}

ViewerApp::~ViewerApp() {
}

void ViewerApp::doUserInput(const std::vector<MinVR::EventRef>& events,
		double synchronizedTime) {
	for (int f = 0; f < events.size(); f++)
	{
		if (events[f]->getName() == "mouse_btn_right_down")
		{
			std::vector<glm::vec3> vertices;

			MinVR::WindowRef window = events[f]->getWindow();
			glm::vec2 res(window->getWidth(), window->getHeight());
			glm::vec2 pos(events[f]->get2DData());
			pos /= res;

			vertices.push_back(glm::vec3(-1.0f, -1.0, 0.0));
			vertices.push_back(glm::vec3(-1.0f, 1.0, 0.0));
			vertices.push_back(glm::vec3(1.0f, 1.0, 0.0));

			vertices.push_back(glm::vec3(1.0f, 1.0, 0.0));
			vertices.push_back(glm::vec3(1.0f, -1.0, 0.0));
			vertices.push_back(glm::vec3(-1.0f, -1.0, 0.0));

			std::vector<unsigned int> indices;
			for (int f = 0; f < vertices.size(); f++)
			{
				vertices[f] += glm::vec3(pos, 0.0f);
				indices.push_back(f);
			}

			_meshes.push_back(vrbase::MeshRef(new vrbase::Mesh(vertices, indices)));
			incrementVersion();
		}
	}

	AppBase::doUserInput(events, synchronizedTime);

}

vrbase::SceneRef ViewerApp::createAppScene(int threadId,
		MinVR::WindowRef window) {

	_mesh->initContext();

	//vrbase::SceneRef scene = vrbase::SceneRef(new vrbase::MeshScene(mesh));
	vrbase::SceneRef scene = vrbase::SceneRef(new vrbase::BlankScene());
	scene = vrbase::SceneRef(new vrbase::BasicRenderedScene(scene));
	return scene;
	//return vrbase::SceneRef(new ViewerAppScene(this));
}

void ViewerApp::drawGraphics(const vrbase::SceneContext& context) {
	_mesh->draw(context);
}

void ViewerApp::preDrawComputation(double synchronizedTime) {
	//_mesh->incrementVersion();
	_mesh->updateImediately();

	AppBase::preDrawComputation(synchronizedTime);
}
