/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef CAMERA_H_
#define CAMERA_H_

#include <glm/glm.hpp>

namespace vrbase {

class Camera {
public:
	virtual ~Camera() {}

	virtual glm::mat4 getProjetionMatrix() = 0;
	virtual glm::mat4 getViewMatrix() = 0;
	virtual glm::mat4 getObjectToWorldMatrix() = 0;
};

} /* namespace vrbase */

#endif /* CAMERA_H_ */
