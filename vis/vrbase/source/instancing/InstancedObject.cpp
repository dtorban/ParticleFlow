/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <vrbase/instancing/InstancedObject.h>

namespace vrbase {

InstancedObject::InstancedObject(VboObjectRef vboObject) : _vboObject(vboObject), _startLocation(0) {
}

InstancedObject::~InstancedObject() {
	cleanupContext();
}

void InstancedObject::initContextItem() {
	_vboObject->initContext();
	_vboObject->generateVaoAttributes(_startLocation);
	initContextItem(_startLocation);
}

bool InstancedObject::updateContextItem(bool changed) {
	if (changed)
	{
		_vboObject->updateContext();
		_vboObject->generateVaoAttributes(_startLocation);
	}

	return updateContextItem(changed, _startLocation);
}

void InstancedObject::draw(const vrbase::SceneContext& context) {
	draw(context, _vboObject);
}

} /* namespace vrbase */
