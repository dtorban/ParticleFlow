/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <vrbase/scenes/SceneContext.h>

namespace vrbase {

SceneContext::SceneContext() : _camera(0), _shader(0) {
}

SceneContext::SceneContext(const SceneContext& context) {
	_camera = context._camera;
	_shader = context._shader;
}

SceneContext::~SceneContext() {
}

} /* namespace vrbase */
