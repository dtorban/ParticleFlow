/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <vrbase/scenes/app/AppScene.h>

namespace vrbase {

AppScene::AppScene(AppBase* app) : _app(app), _lastAppVersion(0) {
}

AppScene::~AppScene() {
}

void AppScene::init() {
	initialize();
	update();
	_lastAppVersion = _app->getVersion();
}

void AppScene::updateFrame() {
	int appVersion =  _app->getVersion();
	if (_lastAppVersion != appVersion)
	{
		update();
		incrementVersion();
		_lastAppVersion = appVersion;
	}
}

} /* namespace vrbase */
