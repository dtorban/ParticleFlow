/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef BASICMOUSELISTENER_H_
#define BASICMOUSELISTENER_H_

#include "vrbase/EventListener.h"

namespace vrbase {

class BasicMouseListener : public EventListener {
public:
	BasicMouseListener(glm::mat4* transformation);
	virtual ~BasicMouseListener();

	void handleEvents(const std::vector<MinVR::EventRef> &events, double synchronizedTime);

private:
	glm::mat4* _transformation;
	bool _rotating;
	glm::vec2 _lastPosition;
};

} /* namespace vrbase */

#endif /* BASICMOUSELISTENER_H_ */
