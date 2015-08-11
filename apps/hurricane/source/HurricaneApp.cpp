/*
 * HurricaneApp.cpp
 *
 *  Created on: Aug 6, 2015
 *      Author: dtorban
 */

#include <HurricaneApp.h>
#include "vrbase/scenes/render/BasicRenderedScene.h"
#include "vrbase/scenes/MeshScene.h"
#include "PFVis/scenes/ParticleScene.h"
#include "PFGpu/partflow/GpuParticleFactory.h"
#include "PFGpu/partflow/emitters/GpuEmitterFactory.h"
#include "PFGpu/partflow/advectors/GpuVectorFieldAdvector.h"
#include "PFCore/partflow/advectors/strategies/RungaKutta4.h"
#include "PFCore/partflow/vectorFields/ConstantField.h"
#include "PFCore/partflow/vectorFields/ParticleFieldVolume.h"

using namespace vrbase;
using namespace PFVis::partflow;
using namespace std;
using namespace PFCore::math;
using namespace PFCore::partflow;

HurricaneApp::HurricaneApp() : PartFlowApp () {
	AppBase::init();

	vector<glm::vec3> vertices;
	//first side
	vertices.push_back(glm::vec3(-1.0f, 0.0, 0.0));
	vertices.push_back(glm::vec3(0.0f, 1.0, 0.0));
	vertices.push_back(glm::vec3(0.0f, 0.0, 1.0));

	vertices.push_back(glm::vec3(0.0f, 0.0, 1.0));
	vertices.push_back(glm::vec3(0.0f, 1.0, 0.0));
	vertices.push_back(glm::vec3(1.0f, 0.0, 0.0));

	vertices.push_back(glm::vec3(0.0f, 0.0, 1.0));
	vertices.push_back(glm::vec3(0.0f, -1.0, 0.0));
	vertices.push_back(glm::vec3(-1.0f, 0.0, 0.0));

	vertices.push_back(glm::vec3(0.0f, 0.0, 1.0));
	vertices.push_back(glm::vec3(1.0f, 0.0, 0.0));
	vertices.push_back(glm::vec3(0.0f, -1.0, 0.0));

	//other side
	vertices.push_back(glm::vec3(1.0f, 0.0, 0.0));
	vertices.push_back(glm::vec3(0.0f, 0.0, -1.0));
	vertices.push_back(glm::vec3(0.0f, -1.0, 0.0));

	vertices.push_back(glm::vec3(0.0f, 0.0, -1.0));
	vertices.push_back(glm::vec3(-1.0f, 0.0, 0.0));
	vertices.push_back(glm::vec3(0.0f, -1.0, 0.0));

	vertices.push_back(glm::vec3(0.0f, 0.0, -1.0));
	vertices.push_back(glm::vec3(1.0f, 0.0, 0.0));
	vertices.push_back(glm::vec3(0.0f, 1.0, 0.0));

	vertices.push_back(glm::vec3(0.0f, 0.0, -1.0));
	vertices.push_back(glm::vec3(0.0f, 1.0, 0.0));
	vertices.push_back(glm::vec3(-1.0f, 0.0, 0.0));

	// quad
	/*vertices.push_back(glm::vec3(-1.0f, -1.0, 0.0));
	vertices.push_back(glm::vec3(-1.0f, 1.0, 0.0));
	vertices.push_back(glm::vec3(1.0f, 1.0, 0.0));

	vertices.push_back(glm::vec3(1.0f, 1.0, 0.0));
	vertices.push_back(glm::vec3(1.0f, -1.0, 0.0));
	vertices.push_back(glm::vec3(-1.0f, -1.0, 0.0));*/

	for (int f = 0; f < vertices.size(); f++)
	{
		vertices[f] *= 0.005;

	}

	vector<unsigned int> indices;
	for (int f = 0; f < vertices.size(); f++)
	{
		indices.push_back(f);
	}

	_mesh = MeshRef(new Mesh(vertices, indices));

	//int numParticles = 1024*512;
	int numParticles = 1024*32;
	//int numParticles = 10;

	GpuParticleFactory psetFactory;
	_localSet = psetFactory.createLocalParticleSet(numParticles, 1);
	_deviceSet = psetFactory.createParticleSet(0, numParticles, 1);
	_localField = psetFactory.createLocalParticleField(0, 1, vec4(-1.0,-1.0,-1.0, 0.0), vec4(10.0, 10.0, 2.0, 1.0), vec4(50,50,10,1));
	_deviceField = psetFactory.createParticleField(0, 0, 1, vec4(-1.0,-1.0,-1.0, 0.0), vec4(10.0, 10.0, 2.0, 1.0), vec4(50,50,10,1));

	const vec4& fieldSize = _deviceField->getSize();

	for (int f = 0; f < fieldSize.x*fieldSize.y*fieldSize.z*fieldSize.t; f++)
	{
		_deviceField->getVectors(0)[f] = vec3(1.0,0.0,0.0);
	}
	//_deviceSet = psetFactory.createLocalParticleSet(1024*1024, 1);

	GpuEmitterFactory emitterFactory;
	_emitter = EmitterRef(emitterFactory.createSphereEmitter(vec3(0.0f), 0.5f, 500));

	for (int f = 0; f < _deviceSet->getNumSteps(); f++)
	{
		_emitter->emitParticles(*_deviceSet, f, true);
	}

	_currentStep = 1;
	/*AdvectorRef advector = AdvectorRef(new GpuVectorFieldAdvector<EulerAdvector<ConstantField>, ConstantField>(EulerAdvector<ConstantField>(), ConstantField(vec3(1.0f ,0.0f, 0.0f))));
	float dt = 0.1f;
	for (int f = 0; f < 10; f++)
	{
		advector->advectParticles(*_deviceSet, f, dt*float(f), dt);
	}*/

	// Copy from device
	_localSet->copy(*_deviceSet);
}

HurricaneApp::~HurricaneApp() {
}

SceneRef HurricaneApp::createAppScene(int threadId, MinVR::WindowRef window)
{
	MeshScene* mesh = new MeshScene(_mesh);
	SceneRef scene = SceneRef(mesh);
	scene = SceneRef(new ParticleScene(scene, mesh, &(*_localSet), Box(glm::vec3(-1.0f), glm::vec3(1.0f))));
	scene = SceneRef(new BasicRenderedScene(scene));
	return scene;
}

void HurricaneApp::preDrawComputation(double synchronizedTime) {
	AdvectorRef advector = AdvectorRef(new GpuVectorFieldAdvector<RungaKutta4<ConstantField>, ConstantField>(RungaKutta4<ConstantField>(), ConstantField(vec3(1.0f ,1.0f, -2.0f))));
	AdvectorRef advector2 = AdvectorRef(new GpuVectorFieldAdvector<RungaKutta4<ConstantField>, ConstantField>(RungaKutta4<ConstantField>(), ConstantField(vec3(-1.0f ,1.0f, -2.0f))));
	ParticleSetView set1 = (*_deviceSet).getView().filterBySize(0, _deviceSet->getNumParticles()/2);
	ParticleSetView set2 = (*_deviceSet).getView().filterBySize(_deviceSet->getNumParticles()/2, _deviceSet->getNumParticles()/2);
	float dt = 0.01f;

	AdvectorRef advector3 = AdvectorRef(new GpuVectorFieldAdvector<RungaKutta4<ParticleFieldVolume>,ParticleFieldVolume>(
					RungaKutta4<ParticleFieldVolume>(),
					ParticleFieldVolume(*_deviceField, 0)));

	for (int f = 0; f < 1; f++)
	{
		//advector->advectParticles(*_deviceSet, _currentStep, dt*float(f), dt);
		advector3->advectParticles(*_deviceSet, _currentStep, dt*float(f), dt);
		_emitter->emitParticles(*_deviceSet, _currentStep);
		//advector->advectParticles(set1, _currentStep, dt*float(f), dt);
		//advector2->advectParticles(set2, _currentStep, dt*float(f), dt);
		//_emitter->emitParticles(set1, _currentStep);
		//_emitter->emitParticles(set2, _currentStep);
		_currentStep++;
	}

	// Copy from device
	_localSet->copy(*_deviceSet);

	AppBase::preDrawComputation(synchronizedTime);
}
