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
	// TODO Auto-generated constructor stub

}

BasicMouseListener::~BasicMouseListener() {
	// TODO Auto-generated destructor stub
}

glm::vec3 getArcballVector(int x, int y, int width, int height) {
  glm::vec3 P = glm::vec3(1.0*x/width*2 - 1.0,
			  1.0*y/height*2 - 1.0,
			  0);
  P.y = -P.y;
  float OP_squared = P.x * P.x + P.y * P.y;
  if (OP_squared <= 1*1)
    P.z = sqrt(1*1 - OP_squared);  // Pythagore
  else
    P = glm::normalize(P);  // nearest point
  return P;
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
		else if (_rotating)
		{
			std::cout << events[f]->getName() << " " << _lastPosition.x << " " << _lastPosition.y << std::endl;
			MinVR::WindowRef window = events[f]->getWindow();
			glm::vec2 res(window->getWidth(), window->getHeight());
			glm::vec2 pos(events[f]->get2DData());
			/*glm::vec2 percentMove = (_lastPosition - pos)/res;
			glm::vec2 norm = glm::normalize(percentMove);
			std::cout << norm.x << " " << norm.y << std::endl;
			*(_transformation) = glm::rotate(*(_transformation), -glm::length(percentMove)*360.0f, glm::vec3(norm.y, norm.x, 0.0f));
			//*(_transformation) = glm::rotate(*(_transformation), -percentMove.x*360.0f, glm::vec3(0.0f, 1.0f, 0.0f));
			//*(_transformation) = glm::rotate(*(_transformation), -percentMove.y*360.0f, glm::vec3(1.0f, 0.0f, 0.0f));*/

			glm::vec3 va = getArcballVector(_lastPosition.x, _lastPosition.y, res.x, res.y);
			glm::vec3 vb = getArcballVector(pos.x, pos.y, res.x, res.y);//getArcballVector( cur_mx,  cur_my);
			float angle = acos(std::min(1.0f, glm::dot(va, vb)));
			glm::vec3 axis_in_camera_coord = glm::cross(va, vb);
			glm::mat3 camera2object(1.0f);
			//glm::mat3 camera2object = glm::inverse(glm::mat3(transforms[MODE_CAMERA]) * glm::mat3(*(_transformation)));
			glm::vec3 axis_in_object_coord = camera2object * axis_in_camera_coord;
			*(_transformation) = glm::rotate(*(_transformation), glm::degrees(angle), axis_in_object_coord);


			_lastPosition = glm::vec2(pos);

		}
	}
}

} /* namespace vrbase */
