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
#include "vrbase/VersionedItem.h"
#include "MVRCore/Event.H"
#include "MVRCore/AbstractWindow.H"
#include <thread>
#include <memory>
#include <map>
#include <iostream>
#include <vector>
#include "GL/glew.h"
#include <glm/glm.hpp>
#include "vrbase/Box.h"

namespace vrbase {

#if defined(WIN32)
#define THREAD_LOCAL __declspec(thread)
#else
#define THREAD_LOCAL thread_local
#endif

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

		currentThreadId = context.threadId;
	}

	static void cleanup()
	{
		contextMap.clear();
		currentThreadId = -1;
	}

	static MinVRGraphicsContext getCurrentContext()
	{
		if (currentThreadId >= 0)
		{
			return contextMap[currentThreadId];
		}

		MinVRGraphicsContext context = { -1, NULL };
		return context;
	}

	static std::map<int, MinVRGraphicsContext> contextMap;
	static THREAD_LOCAL int currentThreadId;
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
		if (VrbaseContext::currentThreadId < 0) { return NULL; }

		typedef typename std::map<int, T*>::iterator it_type;
		it_type it = _threadMap.find(VrbaseContext::currentThreadId);
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

		if (VrbaseContext::currentThreadId >= 0)
		{
			_threadMap[VrbaseContext::currentThreadId] = value;
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

class ContextObject : public VersionedItem {
public:
	virtual ~ContextObject() {
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
		if (VrbaseContext::getCurrentContext().threadId < 0)
		{
			std::vector<MinVRGraphicsContext> contexts = _initialized.getContexts();
			for (int f = 0; f < contexts.size(); f++)
			{
				VrbaseContext::currentThreadId = contexts[f].threadId;
				contexts[f].window->makeContextCurrent();
				cleanupContextItem();
				contexts[f].window->releaseContext();
			}

			VrbaseContext::currentThreadId = -1;
		}
	}

	void updateImediately()
	{
		if (VrbaseContext::currentThreadId < 0)
		{
			std::vector<MinVRGraphicsContext> contexts = _initialized.getContexts();
			for (int f = 0; f < contexts.size(); f++)
			{
				VrbaseContext::currentThreadId = contexts[f].threadId;
				contexts[f].window->makeContextCurrent();
				updateContext();
				contexts[f].window->releaseContext();
			}

			VrbaseContext::currentThreadId = -1;
		}
	}

protected:
	virtual void initContextItem() = 0;
	virtual bool updateContextItem(bool changed) = 0;
	virtual void cleanupContextItem() = 0;

private:
	bool isInitialized() { return _initialized.get() != NULL && *_initialized; }
	ContextSpecificPtr<bool> _initialized;
	ContextSpecificPtr<int> _oldVersion;
};

class SceneContext;

class GraphicsObject : public ContextObject {
public:
	virtual ~GraphicsObject() {}

	virtual const Box getBoundingBox() = 0;
	virtual void draw(const vrbase::SceneContext& context) = 0;
};

} /* namespace vrbase */

#endif /* GRAPHICSOBJECT_H_ */
