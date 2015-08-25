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
	BasicRenderedScene(SceneRef scene, ShaderRef shader);
	virtual ~BasicRenderedScene();

	void init();
	virtual void draw(const SceneContext& context);

	void setShader(ShaderRef shader);

protected:
	virtual void setShaderParameters(const Camera& camera, ShaderRef shader);

private:
	ShaderRef _shader;
};

} /* namespace vrbase */

#endif /* BASICRENDEREDSCENE_H_ */
