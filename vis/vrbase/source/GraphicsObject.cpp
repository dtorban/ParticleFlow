/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <vrbase/GraphicsObject.h>

namespace vrbase {

std::map<int, MinVRGraphicsContext> VrbaseContext::contextMap = std::map<int, MinVRGraphicsContext>();
THREAD_LOCAL int VrbaseContext::currentThreadId = -1;
//__declspec(thread) MinVRGraphicsContext VrbaseContext::currentContext = { -1, NULL };

} /* namespace vrbase */
