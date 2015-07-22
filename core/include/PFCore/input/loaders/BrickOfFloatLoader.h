/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef BRICKOFFLOATLOADER_H_
#define BRICKOFFLOATLOADER_H_

#include "PFCore/input/DataLoader.h"
#include <string>

namespace PFCore {
namespace input {

class BrickOfFloatLoader: public DataLoader {
public:
	BrickOfFloatLoader(const std::string &fileName);
	BrickOfFloatLoader(const std::string &fileName, int xsize, int ysize,
			int zsize, int tsize, int skip);
	virtual ~BrickOfFloatLoader();

	bool load(float *data, int size);

private:
	int getLocation(int x, int y, int z, int t);
	std::string _fileName;
	int _xsize, _ysize, _zsize, _tsize, _skip;
};

}}

#endif /* BRICKOFFLOATLOADER_H_ */
