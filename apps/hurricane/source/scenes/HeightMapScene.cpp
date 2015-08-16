/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/lgpl-3.0.html.
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <scenes/HeightMapScene.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "MVRCore/CameraOffAxis.H"
#include <iostream>
#include "PFCore/input/loaders/BrickOfFloatLoader.h"
#include "vrbase/textures/DataTexture.h"
#include "MVRCore/ConfigMap.H"
#include "MVRCore/ConfigVal.H"
#include "vrbase/textures/DataTexture.h"

using namespace vrbase;

HeightMapScene::HeightMapScene(SceneRef shadowScene, float* heightData, std::string shaderDir) : _vao(0), _vbo(0), _indexVbo(0), _shadowScene(shadowScene) {
	MinVR::ConfigMapRef configMap = MinVR::ConfigValMap::map;
	std::string dataDir = configMap->get("DataDir", "Data");

	_shader = ShaderRef(new Shader(shaderDir + "/hurricane/heightmap.vsh", shaderDir + "/hurricane/heightmap.fsh"));
	_shadowShader = ShaderRef(new Shader(shaderDir + "/hurricane/shadow.vsh", shaderDir + "/hurricane/shadow.fsh"));

	_texture = TextureRef(new DataTexture(500,500,&heightData[0]));

	_boundingBox = new Box(glm::vec3(0.0f), glm::vec3(2139.0f, 2004.0f, 198.0f));
}

HeightMapScene::~HeightMapScene() {
	glDeleteVertexArrays(1, &_vao);
	glDeleteBuffers(1, &_vbo);
	glDeleteBuffers(1, &_indexVbo);
	delete _boundingBox;
}

void HeightMapScene::make_plane(int rows, int columns, float *vertices, unsigned int *indices){
    // Set up vertices
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < columns; ++c) {
            int index = r*columns + c;
            vertices[3*index + 0] = (((float) c)/500.0f-0.5);
            vertices[3*index + 1] = (((float) r)/500.0f-0.5);
            vertices[3*index + 2] = 0.0f;
        }
    }

    // Set up indices
    int i = 0;
    for (int r = 0; r < rows - 1; ++r) {
        indices[i++] = r * columns;
        for (int c = 0; c < columns; ++c) {
            indices[i++] = r * columns + c;
            indices[i++] = (r + 1) * columns + c;
        }
        indices[i++] = (r + 1) * columns + (columns - 1);
    }

    std::cout << "IIIIIII " << i << std::endl;
}

void HeightMapScene::init() {
	int numVertices = 500 * 500 * 3;
	int numIndices = 499 * (2 + 500 * 2);

	float* vertices = new float[numVertices];
	unsigned int* indices = new unsigned int[numIndices];

	make_plane(500,500,vertices,indices);

	glGenVertexArrays(1, &_vao);
	glGenBuffers(1, &_vbo);
	glBindVertexArray(_vao);
	glBindBuffer(GL_ARRAY_BUFFER, _vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*numVertices, 0, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*numVertices, vertices);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (char*)0 + 0*sizeof(GLfloat));

    // create indexes
	glGenBuffers(1, &_indexVbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexVbo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*numIndices, 0, GL_STATIC_DRAW);
	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof(unsigned int)*numIndices, indices);

	std::cout << "Size!!!!!!!!! " << sizeof(vertices) << " " << sizeof(indices) << std::endl;

	_shader->init();
	_shadowShader->init();

	delete[] vertices;
	delete[] indices;
}

void HeightMapScene::updateFrame() {
}

void HeightMapScene::draw(const vrbase::Camera& camera) {

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, _texture->getId());



	glm::mat4 projection = glm::mat4(camera.getProjetionMatrix());
	glm::mat4 view = glm::mat4(camera.getViewMatrix());

	/*_shadowShader->useProgram();
	glm::mat4 shadow = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, 0.0f, 1.0f));
	_shader->setParameter("Shadow", shadow);
	_shader->setParameter("Model", objectToWorld);
	_shader->setParameter("View", view);
	_shader->setParameter("Projection", projection);
	_shader->setParameter("height", 0);
	_shader->setParameter("size", getBoundingBox().getHigh());
	_shadowScene->draw(time, camera, window, shadow);*/

	_shader->useProgram();
	//glm::mat4 model(1.0f);
	/*glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0, 0.5, -2.0f));
	model = glm::rotate(model, 45.0f, glm::vec3(1.0f, 0.0f, 0.0f));
	model = glm::scale(model, glm::vec3(1.5f, 1.5f, 1.5f));
	model = glm::scale(model, glm::vec3(4.0f, 4.0f, 4.0f));*/

	glm::mat4 model = camera.getObjectToWorldMatrix();
	model = glm::translate(model, glm::vec3(0.0, 0.0f, -_boundingBox->getHigh().z/2.0));
	model = glm::scale(model, _boundingBox->getHigh());
	model = glm::translate(model, glm::vec3(0.5));

	_shader->setParameter("Model", model);
	_shader->setParameter("View", view);
	_shader->setParameter("Projection", projection);
	_shader->setParameter("height", 0);

	glBindVertexArray(_vao);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexVbo);

	glDrawElements(GL_TRIANGLE_STRIP, 499*(2+500*2), GL_UNSIGNED_INT, 0);
}

const vrbase::Box HeightMapScene::getBoundingBox() {
	return *_boundingBox;
}
