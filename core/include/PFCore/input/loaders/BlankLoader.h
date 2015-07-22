/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef BLANKLOADER_H_
#define BLANKLOADER_H_

#include "PFCore/input/DataLoader.h"

namespace PFCore {
namespace input {

class BlankLoader : public DataLoader {
public:
	BlankLoader(float val = 0);
	virtual ~BlankLoader();

	bool load(float *data, int size);

private:
	float _val;
};

}}

#endif /* BLANKLOADER_H_ */
