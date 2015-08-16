/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <vrbase/events/BasicTouchListener.h>
#include <iostream>

namespace vrbase {

BasicTouchListener::BasicTouchListener(glm::mat4* transformation) : _transformation(transformation) {
}

BasicTouchListener::~BasicTouchListener() {
}

void BasicTouchListener::handleEvents(
		const std::vector<MinVR::EventRef>& events, double synchronizedTime) {

	for (int f = 0; f < events.size(); f++)
	{
		if (events[f]->getName() == "mouse_btn_left_down")
		{
		}
		else {//mouse_btn_middle_up
			//std::cout << events[f]->getName() << std::endl;
		}
	}
}

} /* namespace vrbase */
