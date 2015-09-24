/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef VBOOBJECT_H_
#define VBOOBJECT_H_

#include <memory>

namespace vrbase {

class VboObject;
typedef std::shared_ptr<VboObject> VboObjectRef;

class VboObject {
public:
	virtual ~VboObject() {}

	virtual void initVboContext() = 0;
	virtual void updateVboContext() = 0;
	virtual void generateVaoAttributes(int &location) = 0;
	virtual int bindIndices() = 0;
};

} /* namespace vrbase */

#endif /* VBOOBJECT_H_ */
