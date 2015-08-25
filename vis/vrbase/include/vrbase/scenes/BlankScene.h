/*
 * BlankScene.h
 *
 *  Created on: Aug 6, 2015
 *      Author: dtorban
 */

#ifndef SOURCE_DIRECTORY__VIS_VRBASE_INCLUDE_VRBASE_SCENES_BLANKSCENE_H_
#define SOURCE_DIRECTORY__VIS_VRBASE_INCLUDE_VRBASE_SCENES_BLANKSCENE_H_

#include "vrbase/Scene.h"

namespace vrbase {

class BlankScene : public Scene {
public:
	BlankScene();
	virtual ~BlankScene();

	const Box getBoundingBox();
	void draw(const SceneContext& context);

	static SceneRef instance();
};

}

#endif /* SOURCE_DIRECTORY__VIS_VRBASE_INCLUDE_VRBASE_SCENES_BLANKSCENE_H_ */
