/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <ViewerAppScene.h>
#include "vrbase/scenes/render/BasicRenderedScene.h"
#include "vrbase/scenes/MeshScene.h"

ViewerAppScene::ViewerAppScene(ViewerApp* viewerApp) : vrbase::AppScene(viewerApp), _viewerApp(viewerApp), _scenes(0), _meshManager(&(viewerApp->_meshes)) {
}

ViewerAppScene::~ViewerAppScene() {
}

void ViewerAppScene::initialize() {
	_scenes = vrbase::CompositeSceneRef(new vrbase::CompositeScene());
	_scenes->init();
}

void ViewerAppScene::update() {
	_meshManager.refresh();
	_scenes->clear();
	for (int f = 0; f < _viewerApp->_meshes.size(); f++)
	{
		std::cout << f << std::endl;
		vrbase::SceneRef scene = _meshManager.get(_viewerApp->_meshes[f]);
		scene = vrbase::SceneRef(new vrbase::BasicRenderedScene(scene));
		_scenes->addScene(scene);
	}

	_scenes->init();
}

const vrbase::Box ViewerAppScene::getBoundingBox() {
	return _scenes->getBoundingBox();
}

void ViewerAppScene::draw(const vrbase::SceneContext& context) {
	_scenes->draw(context);
}
