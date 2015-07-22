/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef SCALELOADER_H_
#define SCALELOADER_H_

#include "PFCore/input/DataLoader.h"

namespace PFCore {
namespace input {

class ScaleLoader: public DataLoader {
public:
	ScaleLoader(DataLoaderRef loader, float scale);
	virtual ~ScaleLoader();

	bool load(float *data, int size);

private:
	DataLoaderRef _loader;
	float _scale;
};

}}

#endif /* SCALELOADER_H_ */
