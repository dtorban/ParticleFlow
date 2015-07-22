/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <PFCore/input/loaders/CompositeDataLoader.h>

namespace PFCore {
namespace input {

CompositeDataLoader::CompositeDataLoader(
		const std::vector<DataLoaderRef>& loaders) : _loaders(loaders) {
}

CompositeDataLoader::~CompositeDataLoader() {
}

bool CompositeDataLoader::load(float* data, int size) {
	for (int f = 0; f < _loaders.size(); f++)
	{
		bool valid = _loaders[f]->load(&data[f*size], size);
		if (!valid)
		{
			return false;
		}
	}

	return true;
}

}}
