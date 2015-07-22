/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include "PFCore/input/loaders/BlankLoader.h"
#include <iostream>

namespace PFCore {
namespace input {

BlankLoader::BlankLoader(float val) : _val(val) {
	// TODO Auto-generated constructor stub

}

BlankLoader::~BlankLoader() {
	// TODO Auto-generated destructor stub
}

bool BlankLoader::load(float* data, int size) {
	for (int f = 0; f < size; f++)
	{
		data[f] = _val;

	}

	return true;
}

}}
