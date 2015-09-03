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
		indices.push_back(f);
	}

	_meshes[0] = vrbase::MeshRef(new vrbase::Mesh(vertices, indices));
}

ViewerApp::~ViewerApp() {
}

vrbase::SceneRef ViewerApp::createAppScene(int threadId,
		MinVR::WindowRef window) {
	return vrbase::SceneRef(new ViewerAppScene(this));
}
