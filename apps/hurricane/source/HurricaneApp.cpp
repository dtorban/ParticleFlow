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
#include "PFCore/partflow/advectors/strategies/EulerAdvector.h"
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
#include "vrbase/scenes/BlankScene.h"
#include "vrbase/scenes/CompositeScene.h"
#include "scenes/HeightMapScene.h"
#include "vrbase/scenes/BufferedScene.h"

using namespace vrbase;
using namespace PFVis::partflow;
using namespace std;
using namespace PFCore::input;
using namespace PFCore::math;
using namespace PFCore::partflow;

HurricaneApp::HurricaneApp() : PartFlowApp () {
	AppBase::init();

	_currentStep = 0;
	_currentParticleTime = 0.0f;
}

HurricaneApp::~HurricaneApp() {
}

void HurricaneApp::init(MinVR::ConfigMapRef configMap) {

	PartFlowApp::init(configMap);

	std::string dataDir = configMap->get("DataDir", "../data");
	_shaderDir = configMap->get("ShaderDir", "../shaders");
	int numParticles = configMap->get<int>("NumParticles", 1024*32);
	int numParticleSteps = configMap->get<int>("NumParticleSteps", 1);
	float particleSize = configMap->get("ParticleSize", 10.0f);
	int startTimeStep = configMap->get<int>("StartTimeStep", 0);
	int numTimeSteps = configMap->get<int>("NumTimeSteps", 1);
	int sampleInterval = configMap->get<int>("SampleInterval", 10);
	_iterationsPerAdvect = configMap->get<int>("IterationsPerAdvect", 1);
	_computeThreadId = configMap->get<int>("ComputeThreadId", -1);
	_dt = configMap->get("dt", 1.0f / 60.0f);
	_noCopy = configMap->get<bool>("NoCopy", false);

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
		vertices[f] *= particleSize;
	}

	vector<unsigned int> indices;
	for (int f = 0; f < vertices.size(); f++)
	{
		indices.push_back(f);
	}

	_mesh = MeshRef(new Mesh(vertices, indices));

	GpuParticleFactory psetFactory;
	_localSet = psetFactory.createLocalParticleSet(numParticles, 0, 0, 1, numParticleSteps);
	_deviceSet = psetFactory.createParticleSet(0, numParticles, 0, 0, 1, numParticleSteps);
	//_deviceSet = psetFactory.createLocalParticleSet(numParticles, 0, 0, 1, 1);

	vec4 startField = vec4(0.0f, 0.0f);
	vec4 lenField = vec4(2139.0f, 2004.0f, 198.0f, numTimeSteps*1.0f);
	vec4 sizeField = vec4(500/sampleInterval, 500/sampleInterval, 100/sampleInterval, numTimeSteps);
	_localField = psetFactory.createLocalParticleField(0, 1, startField, lenField, sizeField);
	_deviceField = psetFactory.createParticleField(0, 0, 1, startField, lenField, sizeField);
	//_deviceField = psetFactory.createLocalParticleField(0, 1, startField, lenField, sizeField);

	const vec4& fieldSize = _deviceField->getSize();

	std::cout << fieldSize.x*fieldSize.y*fieldSize.z*fieldSize.t << std::endl;

	GpuEmitterFactory emitterFactory;
	EmitterRef emitter = EmitterRef(emitterFactory.createBoxEmitter(startField, startField + lenField, numParticles));
	_emitters.push_back(emitter);
	//emitter = EmitterRef(emitterFactory.createSphereEmitter(vec3(0.0f), 50, 500));
	//_emitters.push_back(emitter);
	//emitter = EmitterRef(emitterFactory.createSphereEmitter(vec3(200.0f), 100, 500));
	//_emitters.push_back(emitter);

	for (int f = 0; f < _deviceSet->getNumSteps(); f++)
	{
		_emitters[0]->emitParticles(*_deviceSet, f, true);
	}

	_localSet->copy(*_deviceSet);

	for (int f = 0; f < numTimeSteps; f++)
	{
		std::stringstream ss;
		ss << startTimeStep + f;
		DataLoaderRef dataLoader = createVectorLoader(dataDir, ss.str(), sampleInterval);
		dataLoader->load(reinterpret_cast<float*>(&_localField->getVectors(0)[(int)(fieldSize.x*fieldSize.y*fieldSize.z*f)]), fieldSize.x*fieldSize.y*fieldSize.z*1);
	}

	DataLoaderRef height = DataLoaderRef(new BrickOfFloatLoader(dataDir + "/HGTdata.bin"));
	height->load(&_heightData[0], 500*500);

	_currentParticleTime = 0.0f;

	_deviceField->copy(*_localField);

	_updater = ParticleUpdaterRef(new GpuParticleUpdater<ParticleFieldUpdater>(ParticleFieldUpdater(ParticleFieldVolume(*_deviceField, 0))));
	_updater->updateParticles(*_deviceSet, _currentStep, _currentParticleTime);

	_localSet->copy(*_deviceSet);

	//_currentStep = 1;
	_currentParticleTime -= _dt*_iterationsPerAdvect;
}

void HurricaneApp::calculate()
{
	calculateParticleSet(_deviceSet);
}

class ComputeScene : public vrbase::SceneAdapter
{
public:
	ComputeScene(HurricaneApp* app) : SceneAdapter(BlankScene::instance()), _app(app) {}
	~ComputeScene() {}

	void updateFrame() 
	{
		_app->calculate();
	}

private:
	HurricaneApp* _app;
};

SceneRef HurricaneApp::createAppScene(int threadId, MinVR::WindowRef window)
{

	int max_attribs;
	glGetIntegerv (GL_MAX_VERTEX_ATTRIBS, &max_attribs);
	int max_other;
	glGetIntegerv (GL_MAX_VARYING_COMPONENTS, &max_other);
	std::cout << "Varying: " << max_attribs << " " << max_other << std::endl;//>

	if (threadId == _computeThreadId)
	{
		return SceneRef(new ComputeScene(this));
	}
	else
	{
		int numTimeSteps = 1;

		vec4 startField = vec4(0.0f, 0.0f);
		vec4 lenField = vec4(2139.0f, 2004.0f, 198.0f, numTimeSteps);

		CompositeScene* world = new CompositeScene();

		MeshScene* mesh = new MeshScene(_mesh);
		SceneRef meshScene = SceneRef(mesh);

		SceneRef bufferedScenes[2];
		for (int f = 0; f < 2; f++)
		{
			SceneRef scene = SceneRef(new ParticleScene(meshScene,
					mesh,
					&(*_localSet),
					this,
					Box(glm::vec3(startField.x, startField.y, startField.z), glm::vec3(startField.x, startField.y, startField.z) + glm::vec3(lenField.x, lenField.y, lenField.z))
					, _noCopy ? 0 : -1));
			scene = SceneRef(new BasicParticleRenderer(scene, *_localSet, &_currentStep));
			bufferedScenes[f] = scene;
		}

		SceneRef land = SceneRef(new HeightMapScene(NULL, &_heightData[0], _shaderDir));

		world->addScene(land);
		//world->addScene(SceneRef(new BufferedScene(bufferedScenes[0], bufferedScenes[1])));
		world->addScene(bufferedScenes[0]);
		world->addScene(SceneRef(new BasicRenderedScene(meshScene)));

		return SceneRef(world);
	}

}

void HurricaneApp::preDrawComputation(double synchronizedTime) {
	_currentStep++;
	_currentParticleTime += _dt*_iterationsPerAdvect;

	if (_computeThreadId < 0 && !_noCopy)
	{
		calculate();
	}

	//_localSet->copy(_deviceSet->getView().filterByStep(_currentStep-1, 1));
	//_localSet->copy(*_deviceSet);
	if (!_noCopy)
	{
		_localSet->copy(_deviceSet->getView().filterByStep(_currentStep,1));
		//_localSet->copy(*_deviceSet);
	}

	//std::cout << GL_MAX_VERTEX_ATTRIBS << std::endl;


	AppBase::preDrawComputation(synchronizedTime);
}

DataLoaderRef HurricaneApp::createVectorLoader(const std::string &dataDir, const std::string &timeStep, int sampleInterval)
{
	DataLoaderRef u, v, w, b;

	b = DataLoaderRef(new BlankLoader());
	DataLoaderRef c = DataLoaderRef(new BlankLoader(5.0f));

	if (sampleInterval > 1)
	{
		u = DataLoaderRef(new BrickOfFloatLoader(dataDir + "/Uf" + timeStep + ".bin", 500, 500, 100, 0, sampleInterval));
		v = DataLoaderRef(new BrickOfFloatLoader(dataDir + "/Vf" + timeStep + ".bin", 500, 500, 100, 0, sampleInterval));
		w = DataLoaderRef(new BrickOfFloatLoader(dataDir + "/Wf" + timeStep + ".bin", 500, 500, 100, 0, sampleInterval));
	}
	else
	{
		u = DataLoaderRef(new BrickOfFloatLoader(dataDir + "/Uf" + timeStep + ".bin"));
		v = DataLoaderRef(new BrickOfFloatLoader(dataDir + "/Vf" + timeStep + ".bin"));
		w = DataLoaderRef(new BrickOfFloatLoader(dataDir + "/Wf" + timeStep + ".bin"));
	}
	std::vector<DataLoaderRef> uvw;
	uvw.push_back(DataLoaderRef(new ScaleLoader(v, -60.0f*60.0f / 1000.0f)));
	uvw.push_back(DataLoaderRef(new ScaleLoader(u, 60.0f*60.0f / 1000.0f)));
	uvw.push_back(DataLoaderRef(new ScaleLoader(w, 60.0f*60.0f / 1000.0f)));
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

void HurricaneApp::calculateParticleSet(PFCore::partflow::ParticleSetRef particleSet)
{


	/*AdvectorRef advector = AdvectorRef(new GpuVectorFieldAdvector<EulerAdvector<ParticleFieldVolume>,ParticleFieldVolume>(
		EulerAdvector<ParticleFieldVolume>(),
		ParticleFieldVolume(*_deviceField, 0)));*/
	AdvectorRef advector = AdvectorRef(new GpuVectorFieldAdvector<RungaKutta4<ParticleFieldVolume>, ParticleFieldVolume>(
		RungaKutta4<ParticleFieldVolume>(),
		ParticleFieldVolume(*_deviceField, 0)));

	advector->advectParticles(*particleSet, _currentStep, _currentParticleTime, _dt, _iterationsPerAdvect);

	for (int f = 0; f < _emitters.size(); f++)
	{
		int emitterParticles = particleSet->getNumParticles()/_emitters.size();
		ParticleSetView view = particleSet->getView().filterBySize(f*emitterParticles, emitterParticles);
		_emitters[f]->emitParticles(view, _currentStep);
	}

	_updater->updateParticles(*particleSet, _currentStep, _currentParticleTime);

}

void HurricaneApp::updateParticleSet(
		PFCore::partflow::ParticleSetRef particleSet) {
	if (_noCopy)
	{
		calculateParticleSet(particleSet);
	}
	else if (_computeThreadId >= 0)
	{
		particleSet->copy(*_deviceSet);
	}
}
