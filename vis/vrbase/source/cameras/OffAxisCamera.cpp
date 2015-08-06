/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <vrbase/cameras/OffAxisCamera.h>

namespace vrbase {

OffAxisCamera::OffAxisCamera(MinVR::CameraOffAxis& camera, const glm::mat4& objectToWorld) : _camera(camera), _objectToWorld(objectToWorld) {

}

OffAxisCamera::~OffAxisCamera() {
}

glm::mat4 OffAxisCamera::getProjetionMatrix() {
	return glm::mat4(_camera.getLastAppliedProjectionMatrix());
}

glm::mat4 OffAxisCamera::getViewMatrix() {
	return glm::mat4(_camera.getLastAppliedProjectionMatrix());
}

glm::mat4 OffAxisCamera::getObjectToWorldMatrix() {
	return _objectToWorld;
}

} /* namespace vrbase */
