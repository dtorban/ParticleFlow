/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <vrbase/events/BasicMouseListener.h>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "MVRCore/AbstractWindow.H"
#include <math.h>
#include <algorithm>

namespace vrbase {

BasicMouseListener::BasicMouseListener(glm::mat4* transformation) : _transformation(transformation), _rotating(false) {
}

BasicMouseListener::~BasicMouseListener() {
}

void BasicMouseListener::handleEvents(
		const std::vector<MinVR::EventRef>& events, double synchronizedTime) {
	for (int f = 0; f < events.size(); f++)
	{
		if (events[f]->getName() == "mouse_btn_left_down")
		{
			_rotating = true;
			_lastPosition = glm::vec2(events[f]->get2DData());
		}
		else if (events[f]->getName() == "mouse_btn_left_up")
		{
			_rotating = false;
		}
		else if (events[f]->getName() == "mouse_btn_left_up")
		{
			_rotating = false;
		}
		else if (events[f]->getName() == "mouse_pointer")
		{
			if (_rotating)
			{
				std::cout << events[f]->getName() << " " << _lastPosition.x << " " << _lastPosition.y << std::endl;
				MinVR::WindowRef window = events[f]->getWindow();
				glm::vec2 res(window->getWidth(), window->getHeight());
				glm::vec2 pos(events[f]->get2DData());
				glm::vec2 percentMove = (_lastPosition - pos)/res;
				glm::vec2 norm = glm::normalize(percentMove);
				*(_transformation) = glm::rotate(*(_transformation), -glm::length(percentMove)*360.0f, glm::vec3(norm.y, norm.x, 0.0f));

				_lastPosition = glm::vec2(pos);
			}
		}
		else if (events[f]->getName() == "mouse_scroll")
		{
			glm::vec2 scroll(events[f]->get2DData());
			std::cout << scroll.x << " " << scroll.y << std::endl;
			float zoom = scroll.y > 0 ? 1.5 : 1.0f/1.5;
			*(_transformation) = glm::scale(*(_transformation), glm::vec3(zoom));
		}
		else {
			//std::cout << events[f]->getName() << " " << _lastPosition.x << " " << _lastPosition.y << std::endl;
		}
	}
}

} /* namespace vrbase */
