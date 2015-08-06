/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include "AppKit_GLFW/MVREngineGLFW.H"
#include "HurricaneApp.h"

int
main(int argc, char** argv)
{
	MinVR::AbstractMVREngine *engine = new MinVR::MVREngineGLFW();
	engine->init(argc, argv);
	MinVR::AbstractMVRAppRef app(new HurricaneApp());
	engine->runApp(app);
	delete engine;
}

