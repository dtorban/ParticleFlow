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
#include "MVRCore/Event.H"
#include "MVRCore/AbstractWindow.H"
#include <thread>
#include <memory>
#include <map>
#include <iostream>
#include <vector>

namespace vrbase {

class GraphicsObject;
typedef std::shared_ptr<GraphicsObject> GraphicsObjectRef;

struct MinVRGraphicsContext
{
	int threadId;
	MinVR::WindowRef window;
};

class VrbaseContext
{
public:
	static void setCurrentContext(MinVRGraphicsContext context)
	{
		if (context.threadId >= 0)
		{
			contextMap[context.threadId] = context;
		}

		currentContext = context;
	}

	static void cleanup()
	{
		contextMap.clear();
		currentContext = {-1, NULL};
	}

	static std::map<int, MinVRGraphicsContext> contextMap;
	static thread_local MinVRGraphicsContext currentContext;
};

template<typename T>
class ContextSpecificPtr {
public:
	ContextSpecificPtr() {}
	ContextSpecificPtr(T* value) { reset(value); }
	virtual ~ContextSpecificPtr() {
		typedef typename std::map<int, T*>::iterator it_type;
		for(it_type iterator = _threadMap.begin(); iterator != _threadMap.end(); iterator++) {
			delete iterator->second;
		}
	}

	T* get()
	{
		if (VrbaseContext::currentContext.threadId < 0) { return NULL; }

		typedef typename std::map<int, T*>::iterator it_type;
		it_type it = _threadMap.find(VrbaseContext::currentContext.threadId);
		if (it != _threadMap.end())
		{
			return it->second;
		}

		return NULL;
	}

	void reset(T* value)
	{
		T* val = get();
		if (val != NULL)
		{
			delete val;
		}

		if (VrbaseContext::currentContext.threadId >= 0)
		{
			_threadMap[VrbaseContext::currentContext.threadId] = value;
		}
	}

	T& operator*() {return *get();}
	T* operator->() {return get();}

	std::vector<MinVRGraphicsContext> getContexts()
	{
		std::vector<MinVRGraphicsContext> contexts;

		typedef typename std::map<int, T*>::iterator it_type;
		for(it_type iterator = _threadMap.begin(); iterator != _threadMap.end(); iterator++) {
			contexts.push_back(VrbaseContext::contextMap[iterator->first]);
		}

		return contexts;
	}

private:
	std::map<int, T*> _threadMap;
};

class GraphicsObject : public VersionedItem {
public:
	virtual ~GraphicsObject() {
	}

	void initContext()
	{
		if (!isInitialized())
		{
			initContextItem();
			_oldVersion.reset(new int(getVersion()));
			_initialized.reset(new bool(true));
		}
	}

	void updateContext()
	{
		if (!isInitialized())
		{
			initContext();
		}

		int version = getVersion();
		if (updateContextItem(*_oldVersion != version))
		{
			*_oldVersion = version;
		}
	}

	void cleanupContext()
	{
		if (VrbaseContext::currentContext.threadId < 0)
		{
			std::vector<MinVRGraphicsContext> contexts = _initialized.getContexts();
			for (int f = 0; f < contexts.size(); f++)
			{
				VrbaseContext::currentContext = contexts[f];
				contexts[f].window->makeContextCurrent();
				cleanupContextItem();
				contexts[f].window->releaseContext();
			}

			VrbaseContext::currentContext = {-1, NULL};
		}
	}

	void updateImediately()
	{
		if (VrbaseContext::currentContext.threadId < 0)
		{
			std::vector<MinVRGraphicsContext> contexts = _initialized.getContexts();
			for (int f = 0; f < contexts.size(); f++)
			{
				VrbaseContext::currentContext = contexts[f];
				contexts[f].window->makeContextCurrent();
				updateContext();
				contexts[f].window->releaseContext();
			}

			VrbaseContext::currentContext = {-1, NULL};
		}
	}

	virtual void draw(const SceneContext& context) = 0;

protected:
	virtual void initContextItem() {}
	virtual bool updateContextItem(bool changed) { return true; }
	virtual void cleanupContextItem() {}

private:
	bool isInitialized() { return _initialized.get() != NULL && *_initialized; }
	ContextSpecificPtr<bool> _initialized;
	ContextSpecificPtr<int> _oldVersion;
};

} /* namespace vrbase */

#endif /* GRAPHICSOBJECT_H_ */
