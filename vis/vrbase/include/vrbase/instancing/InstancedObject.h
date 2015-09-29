/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef INSTANCEDOBJECT_H_
#define INSTANCEDOBJECT_H_

#include "vrbase/VboObject.h"
#include "vrbase/GraphicsObject.h"

namespace vrbase {

class InstancedObject : public GraphicsObject {
public:
	InstancedObject(VboObjectRef vboObject);
	virtual ~InstancedObject();

	void initContextItem();
	bool updateContextItem(bool changed);
	void draw(const vrbase::SceneContext& context);

	virtual bool initContextItem(int startLocation) = 0;
	virtual bool updateContextItem(bool changed, int startLocation) = 0;
	virtual void draw(const vrbase::SceneContext& context, VboObjectRef vboObject) = 0;

	void setVboObject(VboObjectRef vboObject) {
		_vboObject = vboObject;
		incrementVersion();
	}

private:
	VboObjectRef _vboObject;
	int _startLocation;
};

} /* namespace vrbase */

#endif /* INSTANCEDOBJECT_H_ */
