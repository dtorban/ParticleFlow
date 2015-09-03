/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <viewer/include/ViewerApp.h>

#include "ViewerAppScene.h"

ViewerApp::ViewerApp() {
}

ViewerApp::~ViewerApp() {
}

vrbase::SceneRef ViewerApp::createAppScene(int threadId,
		MinVR::WindowRef window) {
	return vrbase::SceneRef(new ViewerAppScene(this));
}
