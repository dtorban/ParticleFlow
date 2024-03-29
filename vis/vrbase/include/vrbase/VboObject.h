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
#include "vrbase/GraphicsObject.h"

namespace vrbase {

class VboObject;
typedef std::shared_ptr<VboObject> VboObjectRef;

class VboObject : public GraphicsObject {
public:
	virtual ~VboObject() {}

	virtual void generateVaoAttributes(int &location) = 0;
	virtual int bindIndices() = 0;
};

} /* namespace vrbase */

#endif /* VBOOBJECT_H_ */
