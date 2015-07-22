/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include "PFCore/input/loaders/VectorLoader.h"

namespace PFCore {
namespace input {

VectorLoader::VectorLoader(const std::vector<DataLoaderRef> &loaders) : _loaders(loaders) {
}

VectorLoader::~VectorLoader() {
}

bool VectorLoader::load(float* data, int size)
{
	float *buffer = new float[size];

	int dimension = getDimension();

	for (int f = 0; f < dimension; f++)
	{
		if (!_loaders[f]->load(buffer, size))
		{
			delete [] buffer;
			return false;
		}

		for (int i = 0; i < size; i++)
		{
			data[f + i*dimension] = buffer[i];
		}
	}

	delete [] buffer;

	return true;
}

}}
