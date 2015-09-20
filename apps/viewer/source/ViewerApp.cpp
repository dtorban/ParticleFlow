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


		std::string vs_text =
				"#version 330\n"
				"layout(location = 0) in vec3 pos;\n"
				"layout(location = 1) in vec3 normal;\n"
				//"layout(location = 2) in vec3 loc;\n"
				"uniform mat4 Model;\n"
				"uniform mat4 View;\n"
				"uniform mat4 Projection;\n"
				"\n"
				"out vec3 p;\n"
				"out vec3 v;\n"
				"out vec3 N;\n"
				"\n"
				"void main() {\n"
				"	//p = (Model * vec4(pos,1.0)).xyz;\n"
				"p = pos*25.0;\n"
				"v = (View * Model * vec4(pos,1.0)).xyz;\n"
				"gl_Position = Projection*View*Model*vec4(pos,1.0);\n"
				//"v = (View * Model * vec4(pos+loc,1.0)).xyz;\n"
				//"gl_Position = Projection*View*Model*vec4(pos+loc,1.0);\n"
				"N = normalize((View*Model*vec4(normal,0)).xyz);\n"
				"}\n";
				/*
	    "#version 330\n"
	    "layout(location = 0) in vec3 position;"
	    "layout(location = 1) in vec3 normal;"
	    "uniform mat4 model;"
	    "uniform mat4 view;"
	    "uniform mat4 proj;"
	    "out vec3 fragColor, fragPosition, fragNormal;\n"
	    "void main() {\n"
	    "  fragColor = vec4(1.0,0.0,0.0,1.0);\n"
	    "  gl_Position = proj * view * model * vec4(position,1.0);\n"
	    "  fragPosition = (model * vec4(position,1.0)).xyz;\n"
	    "  fragNormal = normalize((transpose(inverse(model)) * vec4(normal,1.0)).xyz);\n"
	    "}\n";*/

	    std::string fs_text =
	    		"#version 330\n"
	    		"layout(location = 0) out vec4 FragColor;\n"
	    		"\n"
	    		"uniform mat4 View;\n"
	    		"in vec3 N;\n"
	    		"in vec3 v;\n"
	    		"in vec3 p;\n"
	    		"\n"
	    		"void main() {\n"
	    		//"	FragColor = vec4(1.0,0,0,1.0);\n"

	    		"	vec3 lightPos = vec3(0.5, 0.0, 3.0);\n"
	    		"	vec4 color = vec4(1.0,0,0,1.0);\n" //vec4(p.y,0.4,1.0,0.3);\n"
	    		"	vec3 L = normalize(lightPos - v); \n"
	    		"\n"
	    		"    //calculate Ambient Term:\n"
	    		"    vec4 Iamb = vec4(0.2, 0.2, 0.2, 1.0);\n"
	    		"    //calculate Diffuse Term:\n"
	    		"    vec4 Idiff = max(dot(N,-L), 0.0) * vec4(0.7, 0.7, 0.7, 1.0);\n"
	    		"    // write Total Color:\n"
	    		"    FragColor = (Idiff + Iamb) * color;\n"
	    		"}\n";

	    _shader = vrbase::ShaderRef(new vrbase::Shader(vs_text, fs_text));
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

	_shader->initContext();
	_mesh->initContext();

	//vrbase::SceneRef scene = vrbase::SceneRef(new vrbase::MeshScene(mesh));
	vrbase::SceneRef scene = vrbase::SceneRef(new vrbase::BlankScene());
	scene = vrbase::SceneRef(new vrbase::BasicRenderedScene(scene));
	return scene;
	//return vrbase::SceneRef(new ViewerAppScene(this));
}

void ViewerApp::drawAppGraphics(const vrbase::SceneContext& context) {
	_shader->useProgram();
	_shader->setParameter("Model", context.getCamera().getObjectToWorldMatrix());
	_shader->setParameter("View", context.getCamera().getViewMatrix());
	_shader->setParameter("Projection", context.getCamera().getProjetionMatrix());
	_mesh->draw(context);
	_shader->releaseProgram();
}

void ViewerApp::preDrawComputation(double synchronizedTime) {
	//_mesh->incrementVersion();
	_mesh->updateImediately();

	AppBase::preDrawComputation(synchronizedTime);
}
