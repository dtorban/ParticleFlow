/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef BASICRENDEREDSCENE_H_
#define BASICRENDEREDSCENE_H_

#include "vrbase/scenes/SceneAdapter.h"
#include "vrbase/Shader.h"

namespace vrbase {

class BasicRenderedScene : public SceneAdapter {
public:
	BasicRenderedScene(SceneRef scene);
	virtual ~BasicRenderedScene();

	void init();
	virtual void draw(const Camera& camera);

private:
	ShaderRef _shader;
};

} /* namespace vrbase */

#endif /* BASICRENDEREDSCENE_H_ */
