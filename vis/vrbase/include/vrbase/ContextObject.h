/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef CONTEXTOBJECT_H_
#define CONTEXTOBJECT_H_



/*
#include "MVRCore/Event.H"
#include <thread>

namespace vrbase {

class ContextObjectBase;

class ThreadContext {
	friend class ContextObjectBase;
public:
	static void setThreadContext(int threadId, MinVR::WindowRef window) {
		_threadId = threadId;
		_window = window;
	}

private:
	static thread_local int _threadId;
	static thread_local MinVR::WindowRef _window;
};

class ContextObjectBase {
public:
	virtual ~ContextObjectBase() {}

protected:
	int getThreadId() { return ThreadContext::_threadId; }
	MinVR::WindowRef getWindow() { return ThreadContext::_window; }
};

template<typename T>
class ContextObject : public ContextObjectBase {
public:
	ContextObject() {}
	virtual ~ContextObject() {}

private:

};*/

} /* namespace vrbase */

#endif /* CONTEXTOBJECT_H_ */
