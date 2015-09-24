/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef MESH_H_
#define MESH_H_

#include <memory>
#include "vrbase/Box.h"
#include <vector>
#include "vrbase/GraphicsObject.h"

namespace vrbase {

class Mesh;
typedef std::shared_ptr<Mesh> MeshRef;

class Mesh : public GraphicsObject {
public:
	Mesh(const std::vector<glm::vec3> &vertices, const std::vector<unsigned int>& indices);
	Mesh(const std::vector<glm::vec3> &vertices, const std::vector<glm::vec3> &normals, const std::vector<unsigned int>& indices);
	virtual ~Mesh();

	const std::vector<unsigned int>& getIndices() const;
	void setIndices(const std::vector<unsigned int>& indices);
	const std::vector<glm::vec3>& getVertices() const;
	void setVertices(const std::vector<glm::vec3>& vertices);
	const std::vector<glm::vec3>& getNormals() const;
	void setNormals(const std::vector<glm::vec3>& normals);

private:
	void calculateBoundingBox();
	void calculateNormals();
	void init();

	Box* _boundingBox;
	bool _hasNormals;
	std::vector<glm::vec3> _vertices;
	std::vector<glm::vec3> _normals;
	std::vector<unsigned int> _indices;

// Graphics code
public:
	const Box getBoundingBox();
	void draw(const SceneContext& context);
	void initVboContext() { initContext(); }
	void updateVboContext() { updateContext(); }
	void generateVaoAttributes(int &location);
	int bindIndices();

protected:
	void initContextItem();
	bool updateContextItem(bool changed);
	void destroyContextItem();

private:
	void createVBO();
	void deleteVBO();

	ContextSpecificPtr<GLuint> _vao;
	ContextSpecificPtr<GLuint> _vbo;
	ContextSpecificPtr<GLuint> _indexVbo;
	/*GL_CONTEXT_ITEM GLuint _vao;
	GL_CONTEXT_ITEM GLuint _vbo;
	GL_CONTEXT_ITEM GLuint _indexVbo;*/
};

} /* namespace vrbase */

#endif /* MESH_H_ */
