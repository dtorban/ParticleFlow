/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef SHAPEEVENTLISTENER_H_
#define SHAPEEVENTLISTENER_H_

#include "vrbase/EventListener.h"

class ShapeEventListener : public vrbase::EventListener {
public:
	ShapeEventListener(float *shape, int size);
	virtual ~ShapeEventListener();

	void handleEvents(const std::vector<MinVR::EventRef> &events, double synchronizedTime);

private:
	float* _shape;
	int _size;
	bool _shaping;
};


#endif /* SHAPEEVENTLISTENER_H_ */
