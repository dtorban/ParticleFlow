/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef GRAPHICSOBJECT_H_
#define GRAPHICSOBJECT_H_

namespace vrbase {

class GraphicsObject {
public:
	virtual ~GraphicsObject() {}

	virtual void generateVaoAttributes(int &location) = 0;
	virtual int bindIndices() = 0;
};

} /* namespace vrbase */

#endif /* GRAPHICSOBJECT_H_ */
