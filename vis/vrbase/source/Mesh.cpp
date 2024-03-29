/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <vrbase/Mesh.h>
#include <iostream>

namespace vrbase {

Mesh::Mesh(const std::vector<glm::vec3>& vertices,
		const std::vector<unsigned int>& indices) : _vertices(vertices), _indices(indices), _boundingBox(0), _hasNormals(false) {
	init();
}

Mesh::Mesh(const std::vector<glm::vec3>& vertices, const std::vector<glm::vec3> &normals,
		const std::vector<unsigned int>& indices) : _vertices(vertices), _normals(normals), _indices(indices), _boundingBox(0), _hasNormals(true) {
	init();
}

Mesh::~Mesh() {
	delete _boundingBox;
	cleanupContext();
}

void Mesh::init() {
	_boundingBox = new Box();
	calculateBoundingBox();
	calculateNormals();
}

const std::vector<unsigned int>& Mesh::getIndices() const {
	return _indices;
}

void Mesh::setIndices(const std::vector<unsigned int>& indices) {
	_indices = indices;
	incrementVersion();
}

const std::vector<glm::vec3>& Mesh::getVertices() const {
	return _vertices;
}

void Mesh::setVertices(const std::vector<glm::vec3>& vertices) {
	_vertices = vertices;
	calculateNormals();
	calculateBoundingBox();
	incrementVersion();
}

const std::vector<glm::vec3>& Mesh::getNormals() const
{
	return _normals;
}

void Mesh::setNormals(const std::vector<glm::vec3>& normals)
{
	_normals = normals;
	_hasNormals = true;
	incrementVersion();
}

const Box Mesh::getBoundingBox() {
	return *_boundingBox;
}

void Mesh::calculateBoundingBox() {
	glm::vec3 min, max;
	for (int f = 0; f < _vertices.size(); f++)
	{
		const glm::vec3& vert = _vertices[f];

		if (f == 0)
		{
			min = vert;
			max = vert;
		}
		else
		{
			if (vert.x < min.x) { min.x = vert.x; }
			if (vert.y < min.y) { min.y = vert.y; }
			if (vert.z < min.z) { min.z = vert.z; }
			if (vert.x > max.x) { max.x = vert.x; }
			if (vert.y > max.y) { max.y = vert.y; }
			if (vert.z > max.z) { max.z = vert.z; }
		}
	}

	_boundingBox->setLow(min);
	_boundingBox->setHigh(max);
}

void Mesh::calculateNormals() {
	if (!_hasNormals)
	{
		_normals.clear();
		std::cout << "NORMS!!!!!!!!!!!!!!" << std::endl;

		for (int f = 0; f < _vertices.size(); f+=3)
		{
			glm::vec3& a = _vertices[f];
			glm::vec3& b = _vertices[f+1];
			glm::vec3& c = _vertices[f+2];
			glm::vec3 cr = glm::cross(c - a, c - b);
			//cr = glm::vec3(1.0f,0,0);
			std::cout << f << std::endl;

			for (int i = 0; i < 3; i++)
			{
				std::cout << " "<< cr.x << " " << cr.y << " " << cr.z << " " << std::endl;
				if (cr != glm::vec3(0.0f))
				{
					_normals.push_back(glm::normalize(cr));
				}
				else
				{
					_normals.push_back(cr);
				}
			}
		}
	}
}

// Graphics code

//GL_CONTEXT_ITEM_INIT GLuint Mesh::_vao = 0;
//GL_CONTEXT_ITEM_INIT GLuint Mesh::_vbo = 0;
//GL_CONTEXT_ITEM_INIT GLuint Mesh::_indexVbo = 0;

void Mesh::createVBO() {
	std::cout << "Update VBO" << std::endl;
	const std::vector<glm::vec3>& vertices = _vertices;
	const std::vector<glm::vec3>& normals = _normals;
	const std::vector<unsigned int>& indices = _indices;
	int numNormals = normals.size();

	_vao.reset(new GLuint(0));
	_vbo.reset(new GLuint(0));
	_indexVbo.reset(new GLuint(0));

	glGenVertexArrays(1, _vao.get());
	glGenBuffers(1, _vbo.get());
	glGenBuffers(1, _indexVbo.get());

	int numVertices = vertices.size();
	int numIndices = indices.size();

	glBindBuffer(GL_ARRAY_BUFFER, *_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*numVertices*3 + sizeof(GLfloat)*numNormals*3, 0, GL_DYNAMIC_DRAW);

	glBindVertexArray(*_vao);
	int loc = 0;
	generateVaoAttributes(loc);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *_indexVbo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*numIndices, 0, GL_DYNAMIC_DRAW);

	glm::vec3* verts = new glm::vec3[numVertices];
	glm::vec3* norms = new glm::vec3[numNormals];
	unsigned int* ind = new unsigned int[numIndices];

	std::copy(vertices.begin(), vertices.end(), verts);
	std::copy(normals.begin(), normals.end(), norms);
	std::copy(indices.begin(), indices.end(), ind);

	glBindBuffer(GL_ARRAY_BUFFER, *_vbo);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat)*numVertices*3, verts);
	glBufferSubData(GL_ARRAY_BUFFER, sizeof(GLfloat)*numVertices*3, sizeof(GLfloat)*numNormals*3, norms);

	// create indexes
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *_indexVbo);
	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof(unsigned int)*numIndices, ind);

	delete[] verts;
	delete[] norms;
	delete[] ind;
}

void Mesh::generateVaoAttributes(int& location) {
	glBindBuffer(GL_ARRAY_BUFFER, *_vbo);
	glEnableVertexAttribArray(location);
	glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (char*)0 + 0*sizeof(GLfloat));
	glEnableVertexAttribArray(++location);
	glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (char*)0 + sizeof(GLfloat)*_vertices.size()*3);
}

int Mesh::bindIndices() {
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *_indexVbo);
	return _indices.size();
}

void Mesh::deleteVBO() {
	glDeleteVertexArrays(1, _vao.get());
	glDeleteBuffers(1, _vbo.get());
	glDeleteBuffers(1, _indexVbo.get());
}

void Mesh::draw(const SceneContext& context) {
	glBindVertexArray(*_vao);
	int numIndices = bindIndices();

	glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}

void Mesh::initContextItem() {
	createVBO();
}

bool Mesh::updateContextItem(bool changed) {
	if (changed)
	{
		deleteVBO();
		createVBO();
	}

	return true;
}

void Mesh::cleanupContextItem() {
	deleteVBO();
}


} /* namespace vrbase */
