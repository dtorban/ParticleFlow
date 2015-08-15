/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef BASICTOUCHLISTENER_H_
#define BASICTOUCHLISTENER_H_

#include "vrbase/EventListener.h"

namespace vrbase {

class BasicTouchListener : public EventListener {
public:
	BasicTouchListener(glm::mat4* transformation);
	virtual ~BasicTouchListener();

	void handleEvents(const std::vector<MinVR::EventRef> &events, double synchronizedTime);

private:
	glm::mat4* _transformation;
};

} /* namespace vrbase */

#endif /* BASICTOUCHLISTENER_H_ */
