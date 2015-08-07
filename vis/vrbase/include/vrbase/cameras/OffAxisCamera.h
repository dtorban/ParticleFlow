/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef OFFAXISCAMERA_H_
#define OFFAXISCAMERA_H_

#include "vrbase/Camera.h"
#include "MVRCore/CameraOffAxis.H"

namespace vrbase {

class OffAxisCamera : public Camera {
public:
	OffAxisCamera(MinVR::CameraOffAxis& camera, const glm::mat4& objectToWorld = glm::mat4(1.0f), glm::vec3 position = glm::vec3(0.0f));
	virtual ~OffAxisCamera();

	glm::vec3 getPosition();
	glm::mat4 getProjetionMatrix();
	glm::mat4 getViewMatrix();
	glm::mat4 getObjectToWorldMatrix();

private:
	MinVR::CameraOffAxis& _camera;
	glm::mat4 _objectToWorld;
	glm::vec3 _position;
};

} /* namespace vrbase */

#endif /* OFFAXISCAMERA_H_ */
