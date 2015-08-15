/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/lgpl-3.0.html.
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include "vrbase/scenes/CompositeScene.h"

namespace vrbase {

CompositeScene::CompositeScene(const std::vector<SceneRef> &scenes) : _scenes(scenes) {
}

CompositeScene::~CompositeScene() {
}

void CompositeScene::init() {
	for (int f = 0; f < _scenes.size(); f++)
	{
		_scenes[f]->init();
	}
}

void CompositeScene::updateFrame() {
	for (int f = 0; f < _scenes.size(); f++)
	{
		_scenes[f]->updateFrame();
	}
}

CompositeScene::CompositeScene() {
}

void CompositeScene::addScene(SceneRef scene) {
	_scenes.push_back(scene);
}

const Box CompositeScene::getBoundingBox() {
	Box box;

	for (int f = 0; f < _scenes.size(); f++)
	{
		if (f == 0)
		{
			box = _scenes[f]->getBoundingBox();
		}
		else
		{
			box = box.merge(_scenes[f]->getBoundingBox());
		}
	}

	return box;
}

void CompositeScene::draw(const Camera& camera) {
	for (int f = 0; f < _scenes.size(); f++)
	{
		_scenes[f]->draw(camera);
	}
}

}
