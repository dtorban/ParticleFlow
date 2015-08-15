/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/lgpl-3.0.html.
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef COMPOSITESCENE_H_
#define COMPOSITESCENE_H_

#include "vrbase/Scene.h"
#include <vector>

namespace vrbase {

class CompositeScene : public Scene {
public:
	CompositeScene();
	CompositeScene(const std::vector<SceneRef> &scenes);
	virtual ~CompositeScene();

	void init();
	void updateFrame();

	const Box getBoundingBox();
	void draw(const Camera& camera);

	void addScene(SceneRef scene);

private:
	std::vector<SceneRef> _scenes;
};

}

#endif /* COMPOSITESCENE_H_ */
