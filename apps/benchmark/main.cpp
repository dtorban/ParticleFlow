/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <iostream>
#include "PFCore/math/v3.h"

using namespace PFCore::math;

int main(int argc, char** argv) {

	vec3 test(1.0);
	std::cout << test.x << "," << test.y << "," << test.z << std::endl;

	return 0;
}
