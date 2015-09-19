/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef VIEWERAPP_H_
#define VIEWERAPP_H_

#include "PFVis/PartFlowApp.h"
#include "vrbase/Mesh.h"
#include <vector>

class ViewerAppScene;

class ViewerApp : public PFVis::partflow::PartFlowApp {
	friend class ViewerAppScene;
public:
	ViewerApp();
	virtual ~ViewerApp();

	void doUserInput(const std::vector<MinVR::EventRef> &events, double synchronizedTime);
	vrbase::SceneRef createAppScene(int threadId, MinVR::WindowRef window);
	void drawGraphics(const vrbase::SceneContext& context);
	void preDrawComputation(double synchronizedTime);

private:
	std::vector<vrbase::MeshRef> _meshes;
	vrbase::MeshRef _mesh;
};

#endif /* VIEWERAPP_H_ */
