/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef GRAPHICSOBJECT_H_
#define GRAPHICSOBJECT_H_

#include "vrbase/Box.h"
#include "vrbase/scenes/SceneContext.h"
#include "vrbase/VersionedItem.h"
#include <thread>
#include <memory>

namespace vrbase {

#define GL_CONTEXT_ITEM_INIT thread_local
#define GL_CONTEXT_ITEM static GL_CONTEXT_ITEM_INIT

class GraphicsObject;
typedef std::shared_ptr<GraphicsObject> GraphicsObjectRef;

class GraphicsObject : public VersionedItem {
public:

	virtual ~GraphicsObject() {
		if (_initialized)
		{
			destroyContextItem();
		}
	}

	void initContext()
	{
		if (!_initialized)
		{
			initContextItem();
			_oldVersion = getVersion();
		}

		_initialized = true;
	}

	void updateContext()
	{
		int version = getVersion();
		if (updateContextItem(_oldVersion != version))
		{
			_oldVersion = version;
		}
	}

	virtual const Box getBoundingBox() = 0;
	virtual void draw(const SceneContext& context) = 0;

protected:
	virtual void initContextItem() {}
	virtual bool updateContextItem(bool changed) { return true; }
	virtual void destroyContextItem() {}

private:
	bool _initialized = false;
	int _oldVersion;
};

} /* namespace vrbase */

#endif /* GRAPHICSOBJECT_H_ */
