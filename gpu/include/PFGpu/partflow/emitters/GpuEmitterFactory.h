/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef GPUEMITTERFACTORY_H_
#define GPUEMITTERFACTORY_H_

#include "PFCore/partflow/emitters/EmitterFactory.h"

namespace PFCore {
namespace partflow {

#ifdef USE_CUDA
extern "C"
GpuEmitterFactory* createCudaEmitterFactory();
#endif

class GpuEmitterFactory : public EmitterFactory {
public:
	GpuEmitterFactory()
	{
#ifdef USE_CUDA
		_innerFactory = createCudaEmitterFactory();
#endif
	}
	virtual ~GpuEmitterFactory()
	{
#ifdef USE_CUDA
private:
		delete _innerFactory;
#endif
	}

	Emitter* createSphereEmitter(math::vec3 pos, float radius, int duration)
	{
#ifdef USE_CUDA
		_innerFactory->createSphereEmitter(pos, radius, duration);
#else
		return EmitterFactory::createSphereEmitter(pos, radius, duration);
#endif
	}

#ifdef USE_CUDA
private:
	GpuEmitterFactory* _innerFactory;
#endif
};

} /* namespace partflow */
} /* namespace PFCore */

#endif /* GPUEMITTERFACTORY_H_ */
