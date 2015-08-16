/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/lgpl-3.0.html.
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef DATATEXTURE_H_
#define DATATEXTURE_H_

#include "vrbase/Texture.h"

namespace vrbase {


class DataTexture : public Texture {
public:
	DataTexture(int width, int height, float *data);
	DataTexture(int width, int height, int depth, float *data);
	virtual ~DataTexture();
};

}

#endif /* DATATEXTURE_H_ */
