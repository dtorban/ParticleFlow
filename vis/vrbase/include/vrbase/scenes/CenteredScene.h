/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef CENTEREDSCENE_H_
#define CENTEREDSCENE_H_

#include "vrbase/scenes/SceneAdapter.h"

namespace vrbase {

class CenteredScene : public SceneAdapter {
public:
	CenteredScene(SceneRef scene, glm::mat4* transformation = 0);
	virtual ~CenteredScene();

	void draw(const Camera& camera);

private:
	glm::mat4* _transformation;
};

} /* namespace vrbase */

#endif /* CENTEREDSCENE_H_ */
