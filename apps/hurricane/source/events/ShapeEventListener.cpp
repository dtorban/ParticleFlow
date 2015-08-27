/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <events/ShapeEventListener.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "MVRCore/AbstractWindow.H"

ShapeEventListener::ShapeEventListener(float *shape, int size) : _shape(shape), _size(size), _shaping(false) {
	// TODO Auto-generated constructor stub

}

ShapeEventListener::~ShapeEventListener() {
	// TODO Auto-generated destructor stub
}

void ShapeEventListener::handleEvents(
		const std::vector<MinVR::EventRef>& events, double synchronizedTime) {
	for (int f = 0; f < events.size(); f++)
	{
		if (events[f]->getName() == "mouse_btn_right_down")
		{
			_shaping = true;
		}
		else if (events[f]->getName() == "mouse_btn_right_up")
		{
			_shaping = false;
		}

		if (events[f]->getName() == "mouse_pointer" || events[f]->getName() == "mouse_btn_right_down")
		{
			if (_shaping)
			{
				MinVR::WindowRef window = events[f]->getWindow();
				glm::vec2 res(window->getWidth(), window->getHeight());
				glm::vec2 pos(events[f]->get2DData());
				glm::vec2 percentLoc = pos/res;

				_shape[int(percentLoc.x*_size)] = (1.0-percentLoc.y)*2.0;
			}
		}
	}
}
