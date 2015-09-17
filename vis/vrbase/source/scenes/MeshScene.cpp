/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <vrbase/scenes/MeshScene.h>
#include <algorithm>

namespace vrbase {

MeshScene::MeshScene(MeshRef mesh) : _mesh(mesh) {
	_meshVersionId = _mesh->getVersion();
	_vboInitialized = false;
}

MeshScene::~MeshScene() {
	//deleteVBO();
}

void MeshScene::init() {
	_mesh->initContext();
	//updateVBO();
}

void MeshScene::updateFrame() {
	_mesh->updateContext();
	/*if (_meshVersionId != _mesh->getVersion())
	{
		updateVBO();
		_meshVersionId = _mesh->getVersion();
	}*/
}

const Box MeshScene::getBoundingBox() {
	return _mesh->getBoundingBox();
}

int MeshScene::getVersion() const {
	return _mesh->getVersion();
}

void MeshScene::updateVBO() {

/*	const std::vector<glm::vec3>& vertices = _mesh->getVertices();
	const std::vector<glm::vec3>& normals = _mesh->getNormals();
	const std::vector<unsigned int>& indices = _mesh->getIndices();
	int numNormals = normals.size();

	if (!_vboInitialized || vertices.size() != _numVertices || indices.size() != _numIndices)
	{

		deleteVBO();

		glGenVertexArrays(1, &_vao);
		glGenBuffers(1, &_vbo);
		glGenBuffers(1, &_indexVbo);

		_numVertices = vertices.size();
		_numIndices = indices.size();

		glBindBuffer(GL_ARRAY_BUFFER, _vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*_numVertices*3 + sizeof(GLfloat)*numNormals*3, 0, GL_DYNAMIC_DRAW);

		glBindVertexArray(_vao);
		int loc = 0;
		generateVaoAttributes(loc);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexVbo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*_numIndices, 0, GL_DYNAMIC_DRAW);

		_vboInitialized = true;
	}

	glm::vec3* verts = new glm::vec3[_numVertices];
	glm::vec3* norms = new glm::vec3[numNormals];
	unsigned int* ind = new unsigned int[_numIndices];

	std::copy(vertices.begin(), vertices.end(), verts);
	std::copy(normals.begin(), normals.end(), norms);
	std::copy(indices.begin(), indices.end(), ind);

	glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat)*_numVertices*3, verts);
	glBufferSubData(GL_ARRAY_BUFFER, sizeof(GLfloat)*_numVertices*3, sizeof(GLfloat)*numNormals*3, norms);

	// create indexes
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexVbo);
	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof(unsigned int)*_numIndices, ind);

	delete[] verts;
	delete[] norms;
	delete[] ind;*/
}

void MeshScene::generateVaoAttributes(int& location) {
	_mesh->generateVaoAttributes(location);
	/*glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	glEnableVertexAttribArray(location);
	glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (char*)0 + 0*sizeof(GLfloat));
	glEnableVertexAttribArray(++location);
	glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (char*)0 + sizeof(GLfloat)*_numVertices*3);*/
}

int MeshScene::bindIndices() {
	_mesh->bindIndices();
	/*glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexVbo);
	return _numIndices;*/
}

void MeshScene::deleteVBO() {
	/*if (_vboInitialized)
	{
		glDeleteVertexArrays(1, &_vao);
		glDeleteBuffers(1, &_vbo);
		glDeleteBuffers(1, &_indexVbo);
	}*/
}

void  MeshScene::draw(const SceneContext& context) {
	/*glBindVertexArray(_vao);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexVbo);

	glDrawElements(GL_TRIANGLES, _numIndices, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);*/
	_mesh->draw(context);
}

} /* namespace vrbase */


