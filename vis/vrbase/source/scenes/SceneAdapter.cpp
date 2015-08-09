/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <vrbase/scenes/SceneAdapter.h>

namespace vrbase {

SceneAdapter::SceneAdapter(SceneRef scene) : _scene(scene) {
}

SceneAdapter::~SceneAdapter() {
}

void SceneAdapter::init() {
	_scene->init();
}

void SceneAdapter::updateFrame() {
	_scene->updateFrame();
}

int SceneAdapter::getVersion() const {
	return _scene->getVersion();
}

const Box SceneAdapter::getBoundingBox() {
	return _scene->getBoundingBox();
}

void SceneAdapter::draw(const Camera& camera) {
	return _scene->draw(camera);
}

SceneRef SceneAdapter::getInnerScene()
{
	return _scene;
}

} /* namespace vrbase */
