/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef BUFFEREDSCENE_H_
#define BUFFEREDSCENE_H_

#include "vrbase/Scene.h"

namespace vrbase {

class BufferedScene : public Scene {
public:
	BufferedScene(SceneRef scene1, SceneRef scene2);
	virtual ~BufferedScene();

	void init();
	void updateFrame();

	const Box getBoundingBox();
	void draw(const Camera& camera);

private:
	SceneRef _scenes[2];
	int _currentBuffer;
};

} /* namespace vrbase */

#endif /* BUFFEREDSCENE_H_ */
