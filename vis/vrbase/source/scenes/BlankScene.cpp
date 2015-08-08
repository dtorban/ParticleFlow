/*
 * BlankScene.cpp
 *
 *  Created on: Aug 6, 2015
 *      Author: dtorban
 */

#include <vrbase/scenes/BlankScene.h>

namespace vrbase {

BlankScene::BlankScene() {
}

BlankScene::~BlankScene() {
}

const Box BlankScene::getBoundingBox() {
	return Box();
}

void BlankScene::draw(const Camera& camera) {
}

SceneRef BlankScene::instance() {
	static SceneRef instance(new BlankScene());
	return instance;
}

}

