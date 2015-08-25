/*
 * Copyright Regents of the University of Minnesota, 2014.  This software is released under the following license: http://opensource.org/licenses/lgpl-3.0.html.
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef SCENE_H_
#define SCENE_H_

#include "vrbase/Box.h"
#include "vrbase/scenes/SceneContext.h"
#include <memory>

namespace vrbase {

class Scene;
typedef std::shared_ptr<class Scene> SceneRef;

class Scene {
public:
	virtual ~Scene() {}

	virtual void init() {}
	virtual void updateFrame() {}
	virtual int getVersion() const { return 0; }

	virtual const Box getBoundingBox() = 0;
	virtual void draw(const SceneContext& context) = 0;

};

}
#endif /* SCENE_H_ */
