/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <vrbase/scenes/CenteredScene.h>
#include "vrbase/cameras/WorldCamera.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace vrbase {

CenteredScene::CenteredScene(SceneRef scene, glm::mat4* transformation) : SceneAdapter(scene), _transformation(transformation) {
}

CenteredScene::~CenteredScene() {
	// TODO Auto-generated destructor stub
}

void CenteredScene::draw(const Camera& camera) {
	glm::mat4 objectToWorld = camera.getObjectToWorldMatrix();

	const Box box = getInnerScene()->getBoundingBox();
	float size = glm::length((box.getHigh()-box.getLow()));

	if (_transformation != NULL)
	{
		objectToWorld *= *(_transformation);
	}

	objectToWorld = glm::scale(objectToWorld, glm::vec3(1.0f/size));
	objectToWorld = glm::translate(objectToWorld, -box.center());

	getInnerScene()->draw(WorldCamera(camera, objectToWorld));
}

} /* namespace vrbase */
