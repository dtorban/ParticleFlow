/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <PFVis/scenes/ParticleScene.h>

namespace PFVis {
namespace partflow {

ParticleScene::ParticleScene(vrbase::SceneRef scene, vrbase::GraphicsObject* graphicsObject) : vrbase::SceneAdapter(scene), _graphicsObject(graphicsObject),
			_vbo(0), _vao(0) {
}

ParticleScene::~ParticleScene() {
	glDeleteVertexArrays(1, &_vao);
	glDeleteBuffers(1, &_vbo);
}

void ParticleScene::init() {
	int numInstances = 5;

	getInnerScene()->init();

	glGenVertexArrays(1, &_vao);
	glGenBuffers(1, &_vbo);

	glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*numInstances*3, 0, GL_DYNAMIC_DRAW);

	int loc = 0;
	glBindVertexArray(_vao);
	_graphicsObject->generateVaoAttributes(loc);
	glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	glEnableVertexAttribArray(++loc);
	glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (char*)0 + 0*sizeof(GLfloat));
	glVertexAttribDivisorARB(loc, 1);
	glBindVertexArray(0);

	glm::vec3* verts = new glm::vec3[numInstances];
	verts[0] = glm::vec3(0.0f);
	verts[1] = glm::vec3(-1.0f, 0, 0);
	verts[2] = glm::vec3(0, -1.0f, 0);
	verts[3] = glm::vec3(1.0f, 0, 0);
	verts[4] = glm::vec3(0, 1.0f, 0);

	glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat)*numInstances*3, verts);

	delete[] verts;
}

void ParticleScene::draw(const vrbase::Camera& camera) {
	glBindVertexArray(_vao);
	int numIndices = _graphicsObject->bindIndices();
	//glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);

	int numInstances = 5;
	glDrawElementsInstancedBaseVertex(GL_TRIANGLES,
			numIndices,
			GL_UNSIGNED_INT,
			(void*)(sizeof(unsigned int) * 0),
			numInstances,
			0);

	glBindVertexArray(0);
}

}}
