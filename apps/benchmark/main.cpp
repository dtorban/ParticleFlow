/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <iostream>
#include "PFCore/math/v3.h"
#include "PFCore/input/loaders/BlankLoader.h"

using namespace PFCore::math;
using namespace PFCore::input;
using namespace std;

int main(int argc, char** argv) {

	BlankLoader loader(2.0);
	float data[10];
	loader.load(data,10);
	cout << data[5] << endl;

	vec3 test(1.0);
	cout << test.x << "," << test.y << "," << test.z << endl;

	return 0;
}
