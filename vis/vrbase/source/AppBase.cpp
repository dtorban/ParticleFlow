/*
 * AppBase.cpp
 *
 *  Created on: Aug 6, 2015
 *      Author: dtorban
 */

#include <vis/vrbase/include/vrbase/AppBase.h>
#include "MVRCore/CameraOffAxis.H"
#include "vrbase/cameras/OffAxisCamera.h"

namespace vrbase {

void AppBase::doUserInputAndPreDrawComputation(
		const std::vector<MinVR::EventRef>& events, double synchronizedTime) {
}

void AppBase::initializeContextSpecificVars(int threadId,
		MinVR::WindowRef window) {
	initializeContext(threadId, window);
	_sceneMutex.lock();
	_threadScenes[threadId] = createScene(threadId, window);
	_sceneMutex.unlock();
	_threadScenes[threadId]->init();
}

void AppBase::postInitialization() {
}

void AppBase::perFrameComputation(int threadId, MinVR::WindowRef window) {
	_threadScenes[threadId]->updateFrame();
}

void AppBase::drawGraphics(int threadId, MinVR::AbstractCameraRef camera,
		MinVR::WindowRef window) {
	MinVR::CameraOffAxis* offAxisCamera = dynamic_cast<MinVR::CameraOffAxis*>(camera.get());
	_threadScenes[threadId]->draw(OffAxisCamera(*offAxisCamera));
}

void AppBase::initializeContext(int threadId, MinVR::WindowRef window) {
}

void AppBase::init() {
	init(MinVR::ConfigValMap::map);
}

void AppBase::init(MinVR::ConfigMapRef configMap) {
}

AppBase::AppBase() {

}

AppBase::~AppBase() {
}

} /* namespace vrbase */


