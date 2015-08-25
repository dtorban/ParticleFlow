/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <vrbase/scenes/BufferedScene.h>

namespace vrbase {

BufferedScene::BufferedScene(SceneRef scene1, SceneRef scene2) : _currentBuffer(0) {
	_scenes[0] = scene1;
	_scenes[1] = scene2;
}

BufferedScene::~BufferedScene() {
}

void BufferedScene::init() {
	for (int f = 0; f < 2; f++)
	{
		_scenes[f]->init();
	}
}

void BufferedScene::updateFrame() {
	_scenes[(_currentBuffer+1)%2]->updateFrame();
	_currentBuffer++;
}

const Box BufferedScene::getBoundingBox() {
	return _scenes[_currentBuffer%2]->getBoundingBox();
}

void BufferedScene::draw(const SceneContext& context) {
	_scenes[_currentBuffer%2]->draw(context);
}

} /* namespace vrbase */
