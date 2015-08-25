/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef SCENEADAPTER_H_
#define SCENEADAPTER_H_

#include "vrbase/Scene.h"

namespace vrbase {

class SceneAdapter : public Scene {
public:
	SceneAdapter(SceneRef scene);
	virtual ~SceneAdapter();

	virtual void init();
	virtual void updateFrame();
	virtual int getVersion() const;

	virtual const Box getBoundingBox();
	virtual void draw(const SceneContext& context);

protected:
	SceneRef getInnerScene();

private:
	SceneRef _scene;
};

} /* namespace vrbase */

#endif /* SCENEADAPTER_H_ */
