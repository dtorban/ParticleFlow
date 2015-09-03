/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef VIEWERAPPSCENE_H_
#define VIEWERAPPSCENE_H_

#include "ViewerApp.h"
#include "vrbase/scenes/app/AppScene.h"
#include "vrbase/scenes/CompositeScene.h"

class ViewerAppScene : public vrbase::AppScene {
public:
	ViewerAppScene(ViewerApp* viewerApp);
	virtual ~ViewerAppScene();

	const vrbase::Box getBoundingBox();
	void draw(const vrbase::SceneContext& context);

protected:
	void update();

private:
	ViewerApp* _viewerApp;
	vrbase::CompositeSceneRef _scenes;
};

#endif /* VIEWERAPPSCENE_H_ */
