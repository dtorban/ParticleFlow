/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/lgpl-3.0.html.
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef HEIGHTMAPSCENE_H_
#define HEIGHTMAPSCENE_H_

#include "GL/glew.h"
#include "vrbase/Scene.h"
#include "vrbase/Shader.h"
#include "vrbase/Texture.h"

class HeightMapScene : public vrbase::Scene {
public:
	HeightMapScene(vrbase::SceneRef shadowScene, float* heightData, std::string shaderDir);
	virtual ~HeightMapScene();

	void init();
	void updateFrame();
	const vrbase::Box getBoundingBox();
	void draw(const vrbase::SceneContext& context);


private:
	void make_plane(int rows, int columns, float *vertices, unsigned int *indices);
	GLuint _vao;
	GLuint _vbo;
	GLuint _indexVbo;
	vrbase::ShaderRef _shader;
	vrbase::ShaderRef _shadowShader;
	vrbase::TextureRef _texture;
	vrbase::Box *_boundingBox;

	vrbase::SceneRef _shadowScene;
};


#endif /* HEIGHTMAPSCENE_H_ */
