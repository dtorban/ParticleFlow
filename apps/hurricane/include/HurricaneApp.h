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
#include "PFCore/partflow/PartflowRef.h"

class HurricaneApp : public PFVis::partflow::PartFlowApp {
public:
	HurricaneApp();
	virtual ~HurricaneApp();

	vrbase::SceneRef createAppScene(int threadId, MinVR::WindowRef window);
	void preDrawComputation(double synchronizedTime);

private:
	vrbase::MeshRef _mesh;
	PFCore::partflow::ParticleSetRef _localSet;
	PFCore::partflow::ParticleSetRef _deviceSet;
	PFCore::partflow::ParticleFieldRef _localField;
	PFCore::partflow::ParticleFieldRef _deviceField;
	PFCore::partflow::EmitterRef _emitter;
	int _currentStep;
};

#endif /* SOURCE_DIRECTORY__APPS_HURRICANE_INCLUDE_HURRICANEAPP_H_ */
