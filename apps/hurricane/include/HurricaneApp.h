/*
 * HurricaneApp.h
 *
 *  Created on: Aug 6, 2015
 *      Author: dtorban
 */

#ifndef SOURCE_DIRECTORY__APPS_HURRICANE_INCLUDE_HURRICANEAPP_H_
#define SOURCE_DIRECTORY__APPS_HURRICANE_INCLUDE_HURRICANEAPP_H_

#include "PFVis/PartFlowApp.h"
#include "GL/glew.h"
#include <GLFW/glfw3.h>
#include "vrbase/Mesh.h"

class HurricaneApp : public PFVis::partflow::PartFlowApp {
public:
	HurricaneApp();
	virtual ~HurricaneApp();

	vrbase::SceneRef createAppScene(int threadId, MinVR::WindowRef window);

private:
	vrbase::MeshRef _mesh;
};

#endif /* SOURCE_DIRECTORY__APPS_HURRICANE_INCLUDE_HURRICANEAPP_H_ */
