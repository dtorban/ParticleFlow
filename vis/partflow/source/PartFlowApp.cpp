/*
 * PartFlowApp.cpp
 *
 *  Created on: Aug 6, 2015
 *      Author: dtorban
 */

#include <PFVis/PartFlowApp.h>
#include "GL/glew.h"
#include "vrbase/EventListener.h"
#include "vrbase/events/BasicMouseListener.h"
#include "vrbase/events/BasicTouchListener.h"
#include "vrbase/scenes/CenteredScene.h"
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "vrbase/cameras/WorldCamera.h"

namespace PFVis {
namespace partflow {

PartFlowApp::PartFlowApp() : vrbase::AppBase(), _objectToWorld(1.0f) {
}

PartFlowApp::~PartFlowApp() {
}

void PartFlowApp::init(MinVR::ConfigMapRef configMap) {
	glm::dvec3 scale = configMap->get("DefaultSceneScale", glm::dvec3(1.0));
	float rotationAngle = configMap->get<float>("DefaultSceneRotation", 0.0);
	glm::dvec3 rotationAxis = configMap->get("DefaultRotationAxis", glm::dvec3(1.0));

	_objectToWorld = glm::scale(_objectToWorld, glm::vec3(scale));
	_objectToWorld = glm::rotate(_objectToWorld, rotationAngle, glm::vec3(rotationAxis));

	addEventListener(vrbase::EventListenerRef(new vrbase::BasicMouseListener(&_objectToWorld)));
	addEventListener(vrbase::EventListenerRef(new vrbase::BasicTouchListener(&_objectToWorld)));
}

void PartFlowApp::initializeContext(int threadId,
		MinVR::WindowRef window) {

	bool _blendMode = false;


	glShadeModel(GL_SMOOTH);                    // shading mathod: GL_SMOOTH or GL_FLAT
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);      // 4-byte pixel alignment

    // enable /disable features
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    //glEnable(GL_CULL_FACE);
	if (!_blendMode)
	{
		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);
	}

     // track material ambient and diffuse from surface color, call it before glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);

	glClearColor(0.f, 0.f, 0.f, 0.f);                   // background color
    glClearStencil(0);                          // clear stencil buffer
    glClearDepth(1.0f);                         // 0 is near, 1 is far
    //glDepthFunc(GL_LESS);
    glDepthFunc(GL_LEQUAL);
    //glBlendFunc(GL_ONE, GL_ONE);				// set the blend function to result = 1*source + 1*destination

	if (_blendMode)
	{
		glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    	glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ZERO);
	}


	GLenum err;
	if((err = glGetError()) != GL_NO_ERROR) {
		std::cout << "GLERROR initGL: "<<err<<std::endl;
	}
}

vrbase::SceneRef PartFlowApp::createScene(int threadId,
		MinVR::WindowRef window) {

	vrbase::SceneRef scene = createAppScene(threadId, window);
	scene = vrbase::SceneRef(new vrbase::CenteredScene(scene, &_objectToWorld));
	return scene;
}

vrbase::SceneRef PartFlowApp::createAppScene(int threadId,
		MinVR::WindowRef window) {
	return vrbase::AppBase::createScene(threadId, window);
}

void PartFlowApp::drawGraphics(const vrbase::SceneContext& context) {
	glm::mat4 objectToWorld = context.getCamera().getObjectToWorldMatrix();

	const vrbase::Box box = getBoundingBox();
	float size = glm::length((box.getHigh()-box.getLow()));

	objectToWorld *= _objectToWorld;

	objectToWorld = glm::scale(objectToWorld, glm::vec3(1.0f/size));
	objectToWorld = glm::translate(objectToWorld, -box.center());

	vrbase::SceneContext newContext(context);
	vrbase::WorldCamera newCamera(context.getCamera(), objectToWorld);
	newContext.setCamera(newCamera);
	drawAppGraphics(newContext);
}

} /* namespace partflow */
} /* namespace PFVis */

