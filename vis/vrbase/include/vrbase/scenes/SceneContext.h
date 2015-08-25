/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef SCENECONTEXT_H_
#define SCENECONTEXT_H_

#include "vrbase/Camera.h"
#include "vrbase/Shader.h"

namespace vrbase {

class SceneContext {
public:
	SceneContext(const SceneContext& context);
	SceneContext();
	virtual ~SceneContext();

	const Camera& getCamera() const {
		return *_camera;
	}

	void setCamera(Camera& camera) {
		_camera = &camera;
	}

	Shader& getShader() const {
		return *_shader;
	}

	void setShader(Shader& shader) {
		_shader = &shader;
	}

private:
	Camera* _camera;
	Shader* _shader;
};

} /* vrbase */

#endif /* SCENECONTEXT_H_ */
