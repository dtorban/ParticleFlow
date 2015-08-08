/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <vrbase/Mesh.h>

namespace vrbase {

Mesh::Mesh(const std::vector<glm::vec3>& vertices,
		const std::vector<int>& indices) : _versionId(0), _vertices(vertices), _indices(indices), _boundingBox(0), _hasNormals(false) {
	init();
}

Mesh::Mesh(const std::vector<glm::vec3>& vertices, const std::vector<glm::vec3> &normals,
		const std::vector<int>& indices) : _versionId(0), _vertices(vertices), _normals(normals), _indices(indices), _boundingBox(0), _hasNormals(true) {
	init();
}

Mesh::~Mesh() {
	delete _boundingBox;
}

void Mesh::init() {
	_boundingBox = new Box();
	calculateBoundingBox();
	calculateNormals();
}

const std::vector<int>& Mesh::getIndices() const {
	return _indices;
}

void Mesh::setIndices(const std::vector<int>& indices) {
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

int Mesh::getVersion() const {
	return _versionId;
}

const Box Mesh::getBoundingBox() const {
	return *_boundingBox;
}

void Mesh::incrementVersion() {
	_versionId++;
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

		for (int f = 0; f < _vertices.size(); f+=3)
		{
			glm::vec3& a = _vertices[f];
			glm::vec3& b = _vertices[f+1];
			glm::vec3& c = _vertices[f+2];
			glm::vec3 cr = glm::cross(c - a, b - a);

			for (int i = 0; i < 3; i++)
			{
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

} /* namespace vrbase */
