/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <PFVis/scenes/ParticleScene.h>
#include <PFGpu/GpuResourceFactory.h>
#include <PFGpu/partflow/GpuParticleFactory.h>
#include <iostream>

namespace PFVis {
namespace partflow {

ParticleScene::ParticleScene(vrbase::SceneRef scene, vrbase::GraphicsObject* graphicsObject, PFCore::partflow::ParticleSetView* particleSet, ParticleSceneUpdater* sceneUpdater, const vrbase::Box& boundingBox, int deviceId) : vrbase::SceneAdapter(scene), _graphicsObject(graphicsObject),
			_vbo(0), _vao(0), _particleSet(particleSet), _sceneUpdater(sceneUpdater), _boundingBox(boundingBox), _deviceId(deviceId), _gpuResource(0) {
}

ParticleScene::~ParticleScene() {
	if (_gpuResource != 0)
	{
		glDeleteVertexArrays(1, &_vao);
		glDeleteBuffers(1, &_vbo);
		delete _gpuResource;
	}
}

void ParticleScene::init() {
	_currentStep = 0;

	int numInstances = _particleSet->getNumParticles()*_particleSet->getNumSteps();

	getInnerScene()->init();

	glGenVertexArrays(1, &_vao);
	glGenBuffers(1, &_vbo);

	glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	glBufferData(GL_ARRAY_BUFFER, numInstances*(sizeof(GLfloat)*3
			+ sizeof(GLint)*_particleSet->getNumAttributes()
			+ sizeof(GLfloat)*_particleSet->getNumValues()
			+ sizeof(GLfloat)*3*_particleSet->getNumVectors()), 0, GL_DYNAMIC_DRAW);

	int loc = 0;
	glBindVertexArray(_vao);
	_graphicsObject->generateVaoAttributes(loc);

	glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	_startLocation = ++loc;
	updateVao();
	/*glEnableVertexAttribArray(_startLocation);
	glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (char*)0 + 0*sizeof(GLfloat)*3);
	glVertexAttribDivisorARB(loc, 1);

	int startPos = numInstances*3*sizeof(GLfloat);

	for (int f = 0; f < _particleSet->getNumAttributes(); f++)
	{
		glEnableVertexAttribArray(++loc);
		glVertexAttribPointer(loc, 1, GL_INT, GL_FALSE, sizeof(GLint), (char*)0 + startPos + sizeof(GLint)*numInstances*f);
		glVertexAttribDivisorARB(loc, 1);
	}

	startPos += sizeof(GLint)*numInstances*_particleSet->getNumAttributes();

	for (int f = 0; f < _particleSet->getNumValues(); f++)
	{
		glEnableVertexAttribArray(++loc);
		glVertexAttribPointer(loc, 1, GL_FLOAT, GL_FALSE, sizeof(GLfloat), (char*)0 + startPos + sizeof(GLfloat)*numInstances*f);
		glVertexAttribDivisorARB(loc, 1);
	}

	startPos += sizeof(GLfloat)*numInstances*_particleSet->getNumValues();

	for (int f = 0; f < _particleSet->getNumVectors(); f++)
	{
		glEnableVertexAttribArray(++loc);
		glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (char*)0 + startPos + sizeof(GLfloat)*3*numInstances*f);
		glVertexAttribDivisorARB(loc, 1);
	}*/

	glBindVertexArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	int startPos = 0;
	glBufferSubData(GL_ARRAY_BUFFER, startPos, sizeof(GLfloat)*numInstances*3, _particleSet->getPositions(0));
	startPos += sizeof(GLfloat)*numInstances*3;
	glBufferSubData(GL_ARRAY_BUFFER, startPos, sizeof(GLint)*numInstances*_particleSet->getNumAttributes(), _particleSet->getAttributes(0));
	startPos += sizeof(GLint)*numInstances*_particleSet->getNumAttributes();
	glBufferSubData(GL_ARRAY_BUFFER, startPos, sizeof(GLfloat)*numInstances*_particleSet->getNumValues(), _particleSet->getValues(0));
	startPos += sizeof(GLfloat)*numInstances*_particleSet->getNumValues();
	glBufferSubData(GL_ARRAY_BUFFER, startPos, sizeof(GLfloat)*numInstances*_particleSet->getNumVectors()*3, _particleSet->getVectors(0));


	PFCore::GpuResourceFactory factory(_deviceId);
	_gpuResource = factory.registerResource(_vbo);

}

void ParticleScene::updateFrame() {
	if (_gpuResource->map())
	{
		PFCore::partflow::GpuParticleFactory factory;
		PFCore::partflow::ParticleSetRef resourceParticleSet = factory.createParticleSet(_gpuResource, _particleSet->getNumParticles(), _particleSet->getNumAttributes(), _particleSet->getNumValues(), _particleSet->getNumVectors(), _particleSet->getNumSteps());

		_sceneUpdater->updateParticleSet(resourceParticleSet);

		_gpuResource->unmap();
	}
	else
	{
		int numInstances = _particleSet->getNumParticles()*_particleSet->getNumSteps();
		glBindBuffer(GL_ARRAY_BUFFER, _vbo);
		int startPos = 0;
		glBufferSubData(GL_ARRAY_BUFFER, startPos, sizeof(GLfloat)*numInstances*3, _particleSet->getPositions(0));
		startPos += sizeof(GLfloat)*numInstances*3;
		glBufferSubData(GL_ARRAY_BUFFER, startPos, sizeof(GLint)*numInstances*_particleSet->getNumAttributes(), _particleSet->getAttributes(0));
		startPos += sizeof(GLint)*numInstances*_particleSet->getNumAttributes();
		glBufferSubData(GL_ARRAY_BUFFER, startPos, sizeof(GLfloat)*numInstances*_particleSet->getNumValues(), _particleSet->getValues(0));
		startPos += sizeof(GLfloat)*numInstances*_particleSet->getNumValues();
		glBufferSubData(GL_ARRAY_BUFFER, startPos, sizeof(GLfloat)*numInstances*_particleSet->getNumVectors()*3, _particleSet->getVectors(0));
	}

	//_currentStep+=2;
	_currentStep++;
}

const vrbase::Box ParticleScene::getBoundingBox() {
	return _boundingBox;
}

void ParticleScene::draw(const vrbase::Camera& camera) {
	glBindVertexArray(_vao);

	updateVao();

	int numIndices = _graphicsObject->bindIndices();

	int numInstances = _particleSet->getNumParticles();
	glDrawElementsInstancedBaseVertex(GL_TRIANGLES,
			numIndices,
			GL_UNSIGNED_INT,
			(void*)(sizeof(unsigned int) * 0),
			numInstances,
			0);

	glBindVertexArray(0);
}

void ParticleScene::updateVao() {
	int loc = _startLocation-1;
	glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	for (int i = 0; i < _particleSet->getNumSteps(); i++)
	{
		glEnableVertexAttribArray(++loc);
		glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (char*)0 + ((_currentStep-i+_particleSet->getNumSteps())%_particleSet->getNumSteps())*_particleSet->getNumParticles()*sizeof(GLfloat)*3);
		glVertexAttribDivisorARB(loc, 1);
	}

	int startPos = _particleSet->getNumSteps()*_particleSet->getNumParticles()*3*sizeof(GLfloat);

	for (int i = 0; i < _particleSet->getNumSteps(); i++)
	{
		for (int f = 0; f < _particleSet->getNumVectors(); f++)
		{
			glEnableVertexAttribArray(++loc);
			glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (char*)0 + startPos + 3*sizeof(GLfloat)*(((_currentStep-i+_particleSet->getNumSteps())%_particleSet->getNumSteps())*_particleSet->getNumParticles()*_particleSet->getNumVectors() + _particleSet->getNumParticles()*f));
			glVertexAttribDivisorARB(loc, 1);
		}
	}


	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

}
}

