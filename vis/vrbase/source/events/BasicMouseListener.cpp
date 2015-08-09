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

namespace vrbase {

BasicMouseListener::BasicMouseListener(glm::mat4* transformation) : _transformation(transformation) {
	// TODO Auto-generated constructor stub

}

BasicMouseListener::~BasicMouseListener() {
	// TODO Auto-generated destructor stub
}

void BasicMouseListener::handleEvents(
		const std::vector<MinVR::EventRef>& events, double synchronizedTime) {
	for (int f = 0; f < events.size(); f++)
	{
		std::cout << events[f]->getName() << std::endl;
		*(_transformation) = glm::rotate(*(_transformation), 10.0f, glm::vec3(1.0f));
	}
}

} /* namespace vrbase */
