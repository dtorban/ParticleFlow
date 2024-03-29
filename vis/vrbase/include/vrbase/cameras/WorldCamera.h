/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef WORLDCAMERA_H_
#define WORLDCAMERA_H_

#include "vrbase/Camera.h"

namespace vrbase {

class WorldCamera : public Camera {
public:
	WorldCamera(const Camera& camera, const glm::mat4& objectToWorld);
	virtual ~WorldCamera();

	glm::vec3 getPosition() const;
	glm::mat4 getProjetionMatrix() const;
	glm::mat4 getViewMatrix() const;
	glm::mat4 getObjectToWorldMatrix() const;

private:
	const Camera& _camera;
	glm::mat4 _objectToWorld;
};

} /* namespace vrbase */

#endif /* WORLDCAMERA_H_ */
