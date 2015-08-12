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
#include "PFCore/input/loaders/BrickOfFloatLoader.h"
#include "PFCore/input/loaders/BlankLoader.h"
#include "PFCore/input/loaders/ScaleLoader.h"
#include "PFCore/input/loaders/VectorLoader.h"
#include "PFCore/input/loaders/CompositeDataLoader.h"
#include "PFVis/scenes/render/BasicParticleRenderer.h"
#include "PFCore/partflow/updaters/strategies/MagnitudeUpdater.h"
#include "PFGpu/partflow/GpuParticleUpdater.h"
#include "PFCore/partflow/updaters/strategies/ParticleFieldUpdater.h"

using namespace vrbase;
using namespace PFVis::partflow;
using namespace std;
using namespace PFCore::input;
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
		//vertices[f] *= 0.005;
		vertices[f] *= 10;
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
	_localSet = psetFactory.createLocalParticleSet(numParticles, 0, 0, 1);
	_deviceSet = psetFactory.createParticleSet(0, numParticles, 0, 0, 1);

/*	_localField = psetFactory.createLocalParticleField(0, 1, vec4(-1.0,-1.0,-1.0, 0.0), vec4(2.0, 2.0, 2.0, 1.0), vec4(500,500,100,1));
	//_deviceField = psetFactory.createLocalParticleField(0, 1, vec4(-1.0,-1.0,-1.0, 0.0), vec4(1.0, 1.0, 1.0, 1.0), vec4(500,500,10,1));
	_deviceField = psetFactory.createParticleField(0, 0, 1, vec4(-1.0,-1.0,-1.0, 0.0), vec4(2.0, 2.0, 2.0, 1.0), vec4(500,500,100,1));*/

	int start = 20;
	int numTimeSteps = 1;

	vec4 startField = vec4(0.0f, 0.0f);
	vec4 lenField = vec4(2139.0f, 2004.0f, 198.0f, 1.0f);//numTimeSteps*60.0f*60.0);
	_localField = psetFactory.createLocalParticleField(0, 1, startField, lenField, vec4(50,50,10,1));
	_deviceField = psetFactory.createParticleField(0, 0, 1, startField, lenField, vec4(50,50,10,1));

	const vec4& fieldSize = _deviceField->getSize();

	std::cout << fieldSize.x*fieldSize.y*fieldSize.z*fieldSize.t << std::endl;

	/*for (int f = 0; f < fieldSize.x*fieldSize.y*fieldSize.z*fieldSize.t; f++)
	{
		_localField->getVectors(0)[f] = vec3(float(std::rand())/RAND_MAX, float(std::rand())/RAND_MAX, float(std::rand())/RAND_MAX) - 0.5;//vec3(1.0,1.0,1.0);
	}*/

	//_deviceField->copy(*_localField);
	//_deviceSet = psetFactory.createLocalParticleSet(1024*1024, 1);

	GpuEmitterFactory emitterFactory;
	//_emitter = EmitterRef(emitterFactory.createSphereEmitter((startField + lenField)/2.0f, PFCore::math::length(vec3(lenField)/2.0f), 500));
	_emitter = EmitterRef(emitterFactory.createBoxEmitter(startField, startField + lenField, 500));
	//_emitter = EmitterRef(emitterFactory.createSphereEmitter(vec3(0.0f), 0.5f, 500));

	for (int f = 0; f < _deviceSet->getNumSteps(); f++)
	{
		_emitter->emitParticles(*_deviceSet, f, true);
	}

	/*AdvectorRef advector = AdvectorRef(new GpuVectorFieldAdvector<EulerAdvector<ConstantField>, ConstantField>(EulerAdvector<ConstantField>(), ConstantField(vec3(1.0f ,0.0f, 0.0f))));
	float dt = 0.1f;
	for (int f = 0; f < 10; f++)
	{
		advector->advectParticles(*_deviceSet, f, dt*float(f), dt);
	}*/

	// Copy from device
	_localSet->copy(*_deviceSet);

	std::stringstream ss;
	ss << start;
	DataLoaderRef dataLoader = createVectorLoader("/home/dan/Data", ss.str(), true);
	dataLoader->load(reinterpret_cast<float*>(_deviceField->getVectors(0)), fieldSize.x*fieldSize.y*fieldSize.z*fieldSize.t);


	//_updater = ParticleUpdaterRef(new GpuParticleUpdater<MagnitudeUpdater>(MagnitudeUpdater(0,0)));
	_updater = ParticleUpdaterRef(new GpuParticleUpdater<ParticleFieldUpdater>(ParticleFieldUpdater(ParticleFieldVolume(*_deviceField, 0))));
	_updater->updateParticles(*_deviceSet, _currentStep);

	_currentStep = 1;
}

HurricaneApp::~HurricaneApp() {
}

SceneRef HurricaneApp::createAppScene(int threadId, MinVR::WindowRef window)
{
	int numTimeSteps = 1;

	vec4 startField = vec4(0.0f, 0.0f);
	vec4 lenField = vec4(2139.0f, 2004.0f, 198.0f, numTimeSteps);

	MeshScene* mesh = new MeshScene(_mesh);
	SceneRef scene = SceneRef(mesh);
	scene = SceneRef(new ParticleScene(scene, mesh, &(*_localSet), Box(glm::vec3(startField.x, startField.y, startField.z), glm::vec3(startField.x, startField.y, startField.z) + glm::vec3(lenField.x, lenField.y, lenField.z))));
	scene = SceneRef(new BasicParticleRenderer(scene));
	return scene;
}

void HurricaneApp::preDrawComputation(double synchronizedTime) {
	AdvectorRef advector = AdvectorRef(new GpuVectorFieldAdvector<RungaKutta4<ConstantField>, ConstantField>(RungaKutta4<ConstantField>(), ConstantField(vec3(1.0f ,1.0f, -2.0f))));
	AdvectorRef advector2 = AdvectorRef(new GpuVectorFieldAdvector<RungaKutta4<ConstantField>, ConstantField>(RungaKutta4<ConstantField>(), ConstantField(vec3(-1.0f ,1.0f, -2.0f))));
	ParticleSetView set1 = (*_deviceSet).getView().filterBySize(0, _deviceSet->getNumParticles()/2);
	ParticleSetView set2 = (*_deviceSet).getView().filterBySize(_deviceSet->getNumParticles()/2, _deviceSet->getNumParticles()/2);
	//float dt = 0.01f;
	float dt = 1.0/60.0f;

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
		_updater->updateParticles(*_deviceSet, _currentStep);
		_currentStep++;
	}

	// Copy from device
	_localSet->copy(*_deviceSet);

	AppBase::preDrawComputation(synchronizedTime);
}

DataLoaderRef HurricaneApp::createVectorLoader(const std::string &dataDir, const std::string &timeStep, bool lowRes)
{
	DataLoaderRef u, v, w, b;

	b = DataLoaderRef(new BlankLoader());
	DataLoaderRef c = DataLoaderRef(new BlankLoader(5.0f));

	if (lowRes)
	{
		u = DataLoaderRef(new BrickOfFloatLoader(dataDir + "/Uf" + timeStep + ".bin", 500, 500, 100, 0, 10));
		v = DataLoaderRef(new BrickOfFloatLoader(dataDir + "/Vf" + timeStep + ".bin", 500, 500, 100, 0, 10));
		w = DataLoaderRef(new BrickOfFloatLoader(dataDir + "/Wf" + timeStep + ".bin", 500, 500, 100, 0, 10));
	}
	else
	{
		u = DataLoaderRef(new BrickOfFloatLoader(dataDir + "/Uf" + timeStep + ".bin"));
		v = DataLoaderRef(new BrickOfFloatLoader(dataDir + "/Vf" + timeStep + ".bin"));
		w = DataLoaderRef(new BrickOfFloatLoader(dataDir + "/Wf" + timeStep + ".bin"));
	}
	std::vector<DataLoaderRef> uvw;
	uvw.push_back(DataLoaderRef(new ScaleLoader(v, -60.0f*60.0f/1000.0f)));
	uvw.push_back(DataLoaderRef(new ScaleLoader(u, 60.0f*60.0f/1000.0f)));
	uvw.push_back(DataLoaderRef(new ScaleLoader(w, 60.0f*60.0f/1000.0f)));
	//uvw.push_back(u);
	//uvw.push_back(v);
	//uvw.push_back(w);
	//uwv.push_back(DataLoaderRef(new ScaleLoader(u, 1.0f)));
	//uwv.push_back(w);
	//uwv.push_back(DataLoaderRef(new ScaleLoader(v, -1.0f)));
	return DataLoaderRef(new VectorLoader(uvw));
}

DataLoaderRef HurricaneApp::createValueLoader(const std::string &dataDir, const std::string &timeStep, const std::vector<std::string>& params)
{
	std::vector<DataLoaderRef> values;
	for (int f = 0; f < params.size(); f++)
	{
		DataLoaderRef val = DataLoaderRef(new BrickOfFloatLoader(dataDir + "/" + params[f] + "f" + timeStep + ".bin"));
		values.push_back(val);
	}

	return DataLoaderRef(new CompositeDataLoader(values));
}
