/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef COMPOSITEDATALOADER_H_
#define COMPOSITEDATALOADER_H_

#include "PFCore/input/DataLoader.h"
#include <vector>

namespace PFCore {
namespace input {

class CompositeDataLoader : public DataLoader {
public:
	CompositeDataLoader(const std::vector<DataLoaderRef> &loaders);
	virtual ~CompositeDataLoader();

	bool load(float* data, int size);
	int getNumLoaders()
	{
		return _loaders.size();
	}

private:
	std::vector<DataLoaderRef> _loaders;
};

}}

#endif /* COMPOSITEDATALOADER_H_ */
