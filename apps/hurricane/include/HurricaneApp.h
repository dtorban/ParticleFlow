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
#include "PFCore/input/DataLoader.h"
#include "PFVis/scenes/update/ParticleSceneUpdater.h"
#include <vector>

class HurricaneApp : public PFVis::partflow::PartFlowApp, public PFVis::partflow::ParticleSceneUpdater {
public:
	HurricaneApp();
	virtual ~HurricaneApp();

	void init(MinVR::ConfigMapRef configMap);
	vrbase::SceneRef createAppScene(int threadId, MinVR::WindowRef window);
	void preDrawComputation(double synchronizedTime);
	void calculate();
	void updateParticleSet(PFCore::partflow::ParticleSetRef particleSet);
	void doUserInput(const std::vector<MinVR::EventRef> &events, double synchronizedTime);
	void asyncComputation();
	void renderingComplete();

private:
	PFCore::input::DataLoaderRef createVectorLoader(const std::string &dataDir, const std::string &timeStep, int sampleInterval = 0);
	PFCore::input::DataLoaderRef createValueLoader(const std::string &dataDir, const std::string &timeStep, const std::vector<std::string>& params);
	void calculateParticleSet(PFCore::partflow::ParticleSetRef particleSet);

	vrbase::MeshRef _mesh;
	PFCore::partflow::ParticleSetRef _localSet;
	PFCore::partflow::ParticleSetRef _deviceSet;
	PFCore::partflow::ParticleFieldRef _localField;
	PFCore::partflow::ParticleFieldRef _deviceField;
	PFCore::partflow::ParticleUpdaterRef _updater;
	int _iterationsPerAdvect;
	int _currentStep;
	float _currentParticleTime;
	int _computeThreadId;
	float _heightData[500*500];
	std::string _shaderDir;
	float _dt;
	bool _noCopy;
	std::vector<PFCore::partflow::EmitterRef> _emitters;
	float* _shape;
};

#endif /* SOURCE_DIRECTORY__APPS_HURRICANE_INCLUDE_HURRICANEAPP_H_ */
