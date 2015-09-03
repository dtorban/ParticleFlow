/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef APPSCENE_H_
#define APPSCENE_H_

#include "vrbase/Scene.h"
#include "vrbase/AppBase.h"

namespace vrbase {

class AppScene : public vrbase::Scene  {
public:
	AppScene(AppBase* app);
	virtual ~AppScene();

	void init();
	void updateFrame();

protected:
	virtual void initialize() = 0;
	virtual void update() = 0;

private:
	AppBase* _app;
	int _lastAppVersion;
};

} /* namespace vrbase */

#endif /* APPSCENE_H_ */
