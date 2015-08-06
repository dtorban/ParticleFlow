/*
 * Copyright Regents of the University of Minnesota, 2014.  This software is released under the following license: http://opensource.org/licenses/lgpl-3.0.html.
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef SCENE_H_
#define SCENE_H_

#include "MVRCore/AbstractCamera.H"
#include "MVRCore/AbstractWindow.H"
#include "vrbase/Box.h"
#include "vrbase/Camera.h"

#include <glm/glm.hpp>

namespace vrbase {

class Scene;
typedef std::shared_ptr<class Scene> SceneRef;

class Scene {
public:
	virtual ~Scene() {}

	virtual void init() = 0;
	virtual void updateFrame() = 0;
	virtual const Box& getBoundingBox() = 0;
	virtual void draw(const Camera& camera) = 0;
};

}
#endif /* SCENE_H_ */
