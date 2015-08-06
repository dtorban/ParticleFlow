/*
 * AppBase.h
 *
 *  Created on: Aug 6, 2015
 *      Author: dtorban
 */

#ifndef SOURCE_DIRECTORY__VIS_VRBASE_INCLUDE_VRBASE_APPBASE_H_
#define SOURCE_DIRECTORY__VIS_VRBASE_INCLUDE_VRBASE_APPBASE_H_

#include "MVRCore/AbstractMVRApp.H"
#include "vrbase/Scene.h"
#include <map>
#include "MVRCore/Thread.h"
#include "MVRCore/ConfigVal.H"

namespace vrbase {

class AppBase : public MinVR::AbstractMVRApp {
public:
	virtual ~AppBase();

	void init();
	void doUserInputAndPreDrawComputation(const std::vector<MinVR::EventRef> &events, double synchronizedTime);
	void initializeContextSpecificVars(int threadId, MinVR::WindowRef window);
	void perFrameComputation(int threadId, MinVR::WindowRef window);
	void drawGraphics(int threadId, MinVR::AbstractCameraRef camera, MinVR::WindowRef window);

	virtual void init(MinVR::ConfigMapRef configMap);
	virtual void postInitialization();
	virtual void initializeContext(int threadId, MinVR::WindowRef window);
	virtual SceneRef createScene(int threadId, MinVR::WindowRef window);

protected:
	AppBase();

private:
	std::map<int, SceneRef> _threadScenes;
	MinVR::Mutex _sceneMutex;

	float _startTime;
	int _numFrames;
};

} /* namespace vrbase */

#endif /* SOURCE_DIRECTORY__VIS_VRBASE_INCLUDE_VRBASE_APPBASE_H_ */
