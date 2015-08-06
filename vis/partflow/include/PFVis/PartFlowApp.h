/*
 * PartFlowApp.h
 *
 *  Created on: Aug 6, 2015
 *      Author: dtorban
 */

#ifndef SOURCE_DIRECTORY__VIS_PARTFLOW_INCLUDE_PFVIS_PARTFLOWAPP_H_
#define SOURCE_DIRECTORY__VIS_PARTFLOW_INCLUDE_PFVIS_PARTFLOWAPP_H_

#include "vrbase/AppBase.h"
//#include "GL/glew.h"
//#include <GLFW/glfw3.h>

namespace PFVis {
namespace partflow {

class PartFlowApp : public vrbase::AppBase {
public:
	virtual ~PartFlowApp();

	virtual void initializeContext(int threadId, MinVR::WindowRef window);

protected:
	PartFlowApp();
};

} /* namespace partflow */
} /* namespace PFVis */

#endif /* SOURCE_DIRECTORY__VIS_PARTFLOW_INCLUDE_PFVIS_PARTFLOWAPP_H_ */
