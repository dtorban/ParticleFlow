/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <vrbase/cameras/WorldCamera.h>

namespace vrbase {

WorldCamera::WorldCamera(Camera& camera, const glm::mat4& objectToWorld) : _camera(camera), _objectToWorld(objectToWorld) {
}

WorldCamera::~WorldCamera() {
	// TODO Auto-generated destructor stub
}

glm::vec3 WorldCamera::getPosition() {
	return _camera.getPosition();
}

glm::mat4 WorldCamera::getProjetionMatrix() {
	return _camera.getProjetionMatrix();
}

glm::mat4 WorldCamera::getViewMatrix() {
	return _camera.getViewMatrix();
}

glm::mat4 WorldCamera::getObjectToWorldMatrix() {
	return _objectToWorld;
}

} /* namespace vrbase */
