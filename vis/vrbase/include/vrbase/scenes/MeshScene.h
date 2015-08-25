/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef MESHSCENE_H_
#define MESHSCENE_H_

#include "GL/glew.h"
#include "vrbase/Scene.h"
#include "vrbase/Mesh.h"
#include "vrbase/GraphicsObject.h"

namespace vrbase {

class MeshScene : public Scene, public GraphicsObject {
public:
	MeshScene(MeshRef mesh);
	virtual ~MeshScene();

	void init();
	void updateFrame();
	int getVersion() const;

	const Box getBoundingBox();
	void draw(const SceneContext& context);

	void generateVaoAttributes(int &location);
	int bindIndices();

private:
	void updateVBO();
	void deleteVBO();

	bool _vboInitialized;
	MeshRef _mesh;
	int _meshVersionId;
	GLuint _vao;
	GLuint _vbo;
	GLuint _indexVbo;
	int _numVertices;
	int _numIndices;
};

} /* namespace vrbase */

#endif /* MESHSCENE_H_ */
