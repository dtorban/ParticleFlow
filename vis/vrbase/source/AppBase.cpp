/*
 * AppBase.cpp
 *
 *  Created on: Aug 6, 2015
 *      Author: dtorban
 */

#include <vis/vrbase/include/vrbase/AppBase.h>
#include "MVRCore/CameraOffAxis.H"
#include "vrbase/cameras/OffAxisCamera.h"
#include "vrbase/scenes/BlankScene.h"
#include "vrbase/GraphicsObject.h"

namespace vrbase {

AppBase::AppBase() : _startTime(-1), _numFrames(0) {

}

AppBase::~AppBase() {
}

void AppBase::doUserInputAndPreDrawComputation(
		const std::vector<MinVR::EventRef>& events, double synchronizedTime) {
	doUserInput(events, synchronizedTime);

	for (int f = 0; f < _eventListeners.size(); f++)
	{
		_eventListeners[f]->handleEvents(events, synchronizedTime);
	}

	preDrawComputation(synchronizedTime);
}

void AppBase::initializeContextSpecificVars(int threadId,
		MinVR::WindowRef window) {
	VrbaseContext::context = {threadId, window};
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

void AppBase::preDrawComputation(double synchronizedTime) {
	if (_startTime < 0)
	{
		_startTime = synchronizedTime;
	}
	else
	{
		_numFrames++;

		float fps = 10 / (synchronizedTime - _startTime);
		if (_numFrames % 10 == 0)
		{
			std::cout << fps << std::endl;
			_startTime = synchronizedTime;
		}
	}
}

void AppBase::drawGraphics(int threadId, MinVR::AbstractCameraRef camera,
		MinVR::WindowRef window) {
	MinVR::CameraOffAxis* offAxisCamera = dynamic_cast<MinVR::CameraOffAxis*>(camera.get());
	SceneContext context;
	OffAxisCamera contextCamera(*offAxisCamera);
	context.setCamera(contextCamera);
	_threadScenes[threadId]->draw(context);
	drawGraphics(context);
}

void AppBase::initializeContext(int threadId, MinVR::WindowRef window) {
}

void AppBase::init() {
	init(MinVR::ConfigValMap::map);
}

void AppBase::init(MinVR::ConfigMapRef configMap) {
}

SceneRef AppBase::createScene(int threadId, MinVR::WindowRef window) {
	return BlankScene::instance();
}

void AppBase::addEventListener(EventListenerRef eventListener) {
	_eventListeners.push_back(eventListener);
}

} /* namespace vrbase */

void vrbase::AppBase::doUserInput(const std::vector<MinVR::EventRef>& events,
		double synchronizedTime) {
}

void vrbase::AppBase::drawGraphics(const SceneContext& context) {
}
