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
#include "vrbase/VboObject.h"
#include "vrbase/scenes/management/ObjectSceneManager.h"

namespace vrbase {

class MeshScene : public Scene, public VboObject {
public:
	MeshScene(MeshRef mesh);
	virtual ~MeshScene();

	void init();
	void updateFrame();
	int getVersion() const;

	const Box getBoundingBox();
	void draw(const SceneContext& context);

	virtual void initVboContext() { init(); }
	virtual void updateVboContext() { updateFrame(); }
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

typedef SceneManager<MeshRef, MeshScene> MeshManager;

} /* namespace vrbase */


#endif /* MESHSCENE_H_ */
