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
	OffAxisCamera(MinVR::CameraOffAxis& camera);
	virtual ~OffAxisCamera();

	glm::mat4 getProjetionMatrix();
	glm::mat4 getViewMatrix();
	glm::mat4 getObjectToWorldMatrix();

private:
	MinVR::CameraOffAxis& _camera;
};

} /* namespace vrbase */

#endif /* OFFAXISCAMERA_H_ */
