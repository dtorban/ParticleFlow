/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef EVENTLISTENER_H_
#define EVENTLISTENER_H_

#include <memory>
#include "MVRCore/Event.H"

namespace vrbase {

class EventListener;
typedef std::shared_ptr<EventListener> EventListenerRef;

class EventListener {
public:
	virtual ~EventListener() {}

	virtual void handleEvents(const std::vector<MinVR::EventRef> &events, double synchronizedTime) = 0;
};

} /* namespace vrbase */

#endif /* EVENTLISTENER_H_ */
