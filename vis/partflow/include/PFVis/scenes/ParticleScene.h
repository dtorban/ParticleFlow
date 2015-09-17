/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef PARTICLESCENE_H_
#define PARTICLESCENE_H_

#include "vrbase/scenes/SceneAdapter.h"
#include "vrbase/VboObject.h"
#include "GL/glew.h"
#include "PFCore/partflow/ParticleSetView.h"
#include "PFGpu/GpuResource.h"
#include "PFVis/scenes/update/ParticleSceneUpdater.h"

namespace PFVis {
namespace partflow {

class ParticleScene : public vrbase::SceneAdapter {
public:
	ParticleScene(vrbase::SceneRef scene, vrbase::VboObject* graphicsObject, PFCore::partflow::ParticleSetView* particleSet, ParticleSceneUpdater* sceneUpdater, const vrbase::Box& boundingBox, int deviceId = -1);
	virtual ~ParticleScene();

	void init();
	void updateFrame();
	const vrbase::Box getBoundingBox();
	void draw(const vrbase::SceneContext& context);

private:
	void updateVao(int positionOffset);

	vrbase::VboObject* _graphicsObject;
	vrbase::Box _boundingBox;

	GLuint _vao[2];
	GLuint _vbo;

	int _deviceId;
	PFVis::partflow::ParticleSceneUpdater* _sceneUpdater;

	PFCore::partflow::ParticleSetView* _particleSet;
	PFCore::GpuResource* _gpuResource;
	int _currentStep;
	int _startLocation;
};

}}

#endif /* PARTICLESCENE_H_ */
