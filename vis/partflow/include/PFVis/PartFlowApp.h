/*
 * PartFlowApp.h
 *
 *  Created on: Aug 6, 2015
 *      Author: dtorban
 */

#ifndef SOURCE_DIRECTORY__VIS_PARTFLOW_INCLUDE_PFVIS_PARTFLOWAPP_H_
#define SOURCE_DIRECTORY__VIS_PARTFLOW_INCLUDE_PFVIS_PARTFLOWAPP_H_

#include "vrbase/AppBase.h"

namespace PFVis {
namespace partflow {

class PartFlowApp : public vrbase::AppBase {
public:
	virtual ~PartFlowApp();

	virtual void init(MinVR::ConfigMapRef configMap);
	virtual void initializeContext(int threadId, MinVR::WindowRef window);
	vrbase::SceneRef createScene(int threadId, MinVR::WindowRef window);
	virtual vrbase::SceneRef createAppScene(int threadId, MinVR::WindowRef window);
	void drawGraphics(const vrbase::SceneContext& context);
	virtual void drawAppGraphics(const vrbase::SceneContext& context) {}
	virtual vrbase::Box getBoundingBox() { return vrbase::Box(); }

protected:
	PartFlowApp();

private:
	glm::mat4 _objectToWorld;
};

} /* namespace partflow */
} /* namespace PFVis */

#endif /* SOURCE_DIRECTORY__VIS_PARTFLOW_INCLUDE_PFVIS_PARTFLOWAPP_H_ */
