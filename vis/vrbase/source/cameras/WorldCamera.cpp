/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <vrbase/cameras/WorldCamera.h>

namespace vrbase {

WorldCamera::WorldCamera(const Camera& camera, const glm::mat4& objectToWorld) : _camera(camera) {
	_objectToWorld = objectToWorld;
}

WorldCamera::~WorldCamera() {
}

glm::vec3 WorldCamera::getPosition() const {
	return _camera.getPosition();
}

glm::mat4 WorldCamera::getProjetionMatrix() const {
	return _camera.getProjetionMatrix();
}

glm::mat4 WorldCamera::getViewMatrix() const {
	return _camera.getViewMatrix();
}

glm::mat4 WorldCamera::getObjectToWorldMatrix() const {
	return _objectToWorld;
}

} /* namespace vrbase */
