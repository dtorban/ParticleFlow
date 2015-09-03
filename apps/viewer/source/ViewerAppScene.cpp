/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <ViewerAppScene.h>

ViewerAppScene::ViewerAppScene(ViewerApp* viewerApp) : vrbase::AppScene(viewerApp), _viewerApp(viewerApp) {
}

ViewerAppScene::~ViewerAppScene() {
}

void ViewerAppScene::update() {
}

const vrbase::Box ViewerAppScene::getBoundingBox() {
	return _scenes->getBoundingBox();
}

void ViewerAppScene::draw(const vrbase::SceneContext& context) {
	_scenes->draw(context);
}
