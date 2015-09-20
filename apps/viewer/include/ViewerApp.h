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
#include "vrbase/Shader.h"
#include <vector>

class ViewerAppScene;

class ViewerApp : public PFVis::partflow::PartFlowApp {
	friend class ViewerAppScene;
public:
	ViewerApp();
	virtual ~ViewerApp();

	void doUserInput(const std::vector<MinVR::EventRef>& events, double synchronizedTime);

	void initializeContext(int threadId, MinVR::WindowRef window);
	void updateContext();
	virtual vrbase::Box getBoundingBox() { return _mesh->getBoundingBox(); }
	void drawAppGraphics(const vrbase::SceneContext& context);

private:
	vrbase::MeshRef _mesh;
	vrbase::ShaderRef _shader;
};

#endif /* VIEWERAPP_H_ */
