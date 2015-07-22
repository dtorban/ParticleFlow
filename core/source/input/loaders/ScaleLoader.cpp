/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <PFCore/input/loaders/ScaleLoader.h>

namespace PFCore {
namespace input {

ScaleLoader::ScaleLoader(DataLoaderRef loader, float scale) : _loader(loader), _scale(scale) {
}

ScaleLoader::~ScaleLoader() {
}

bool ScaleLoader::load(float* data, int size) {
	bool loaded = _loader->load(data, size);

	for (int f = 0; f < size; f++)
	{
		data[f] *= _scale;
	}

	return loaded;
}

}}
