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

ParticleScene::ParticleScene(vrbase::SceneRef scene, vrbase::GraphicsObject* graphicsObject, PFCore::partflow::ParticleSetView* particleSet, ParticleSceneUpdater* sceneUpdater, const vrbase::Box& boundingBox) : vrbase::SceneAdapter(scene), _graphicsObject(graphicsObject),
			_vbo(0), _vao(0), _particleSet(particleSet), _sceneUpdater(sceneUpdater), _boundingBox(boundingBox) {
}

ParticleScene::~ParticleScene() {
	glDeleteVertexArrays(1, &_vao);
	glDeleteBuffers(1, &_vbo);
	delete _gpuResource;
}

void ParticleScene::init() {
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
	glEnableVertexAttribArray(++loc);
	glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (char*)0 + 0*sizeof(GLfloat));
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
		glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (char*)0 + startPos + sizeof(GLfloat)*numInstances*f);
		glVertexAttribDivisorARB(loc, 1);
	}

	glBindVertexArray(0);

	/*
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (char*)0 + 0*sizeof(GLfloat));
    for (int f = 0; f < _numValues; f++)
    {
    	glEnableVertexAttribArray(f+1);
        glVertexAttribPointer(f+1, 1, GL_FLOAT, GL_FALSE, sizeof(GLfloat), (char*)0 + sizeof(glm::vec3)*_particleField->getSize() + (f)*sizeof(GLfloat)*_particleField->getSize());
    }
	 */

	glm::vec3* verts = new glm::vec3[numInstances];
	verts[0] = glm::vec3(0.0f);
	verts[1] = glm::vec3(-1.0f, 0, 0);
	verts[2] = glm::vec3(0, -1.0f, 0);
	verts[3] = glm::vec3(1.0f, 0, 0);
	verts[4] = glm::vec3(0, 1.0f, 0);

	glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	startPos = 0;
	glBufferSubData(GL_ARRAY_BUFFER, startPos, sizeof(GLfloat)*numInstances*3, _particleSet->getPositions(0));
	startPos += sizeof(GLfloat)*numInstances*3;
	glBufferSubData(GL_ARRAY_BUFFER, startPos, sizeof(GLint)*numInstances*_particleSet->getNumAttributes(), _particleSet->getAttributes(0));
	startPos += sizeof(GLint)*numInstances*_particleSet->getNumAttributes();
	glBufferSubData(GL_ARRAY_BUFFER, startPos, sizeof(GLfloat)*numInstances*_particleSet->getNumValues(), _particleSet->getValues(0));
	startPos += sizeof(GLfloat)*numInstances*_particleSet->getNumValues();
	glBufferSubData(GL_ARRAY_BUFFER, startPos, sizeof(GLfloat)*numInstances*_particleSet->getNumVectors()*3, _particleSet->getVectors(0));

	delete[] verts;

	PFCore::GpuResourceFactory factory(0);
	_gpuResource = factory.registerResource(_vbo);
}

void ParticleScene::updateFrame() {
	/*
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
	*/

	//char *data;
	//PFCore::partflow::ParticleSetView pset = *_particleSet;

	if (_gpuResource->map())
	{
		//int size = _gpuResource->getData((void **)(&data));
		//std::cout << size << " ----------------------------------------------------" << _particleSet->getSize() << std::endl;
		//if (size > 0)
		//{
			/*int numInstances = _particleSet->getNumParticles()*_particleSet->getNumSteps();
			int startPos = 0;
			pset.setPositions(reinterpret_cast<PFCore::math::vec3*>(&data[startPos]));
			startPos += sizeof(PFCore::math::vec3)*numInstances;
			pset.setAttributes(reinterpret_cast<int*>(&data[startPos]));
			startPos += sizeof(int)*numInstances*_particleSet->getNumAttributes();
			pset.setValues(reinterpret_cast<float*>(&data[startPos]));
			startPos += sizeof(float)*numInstances*_particleSet->getNumValues();
			pset.setVectors(reinterpret_cast<PFCore::math::vec3*>(&data[startPos]));*/
		//}
		PFCore::partflow::GpuParticleFactory factory;
		PFCore::partflow::ParticleSetRef resourceParticleSet = factory.createParticleSet(_gpuResource, _particleSet->getNumParticles(), _particleSet->getNumAttributes(), _particleSet->getNumValues(), _particleSet->getNumVectors(), _particleSet->getNumSteps());

		//resourceParticleSet->copy(*_particleSet);
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


}

const vrbase::Box ParticleScene::getBoundingBox() {
	return _boundingBox;
}

void ParticleScene::draw(const vrbase::Camera& camera) {
	glBindVertexArray(_vao);
	int numIndices = _graphicsObject->bindIndices();
	//glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);

	int numInstances = _particleSet->getNumParticles()*_particleSet->getNumSteps();
	glDrawElementsInstancedBaseVertex(GL_TRIANGLES,
			numIndices,
			GL_UNSIGNED_INT,
			(void*)(sizeof(unsigned int) * 0),
			numInstances,
			0);

	glBindVertexArray(0);
}

}
}

