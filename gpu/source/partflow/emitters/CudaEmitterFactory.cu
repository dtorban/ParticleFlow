/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <PFGpu/partflow/emitters/CudaEmitterFactory.h>
#include "PFGpu/partflow/CudaEmitter.cuh"

namespace PFCore {
namespace partflow {

CudaEmitterFactory::CudaEmitterFactory(int randSize) : rnd(randSize) {
	// TODO Auto-generated constructor stub

}

CudaEmitterFactory::~CudaEmitterFactory() {
	// TODO Auto-generated destructor stub
}

Emitter* CudaEmitterFactory::createSphereEmitter(math::vec3 pos, float radius, int duration, int deviceId = -1)
{
	return new CudaEmitter<SphereEmitter>(SphereEmitter(pos, radius, duration, rnd));
}

extern "C"
GpuEmitterFactory* createCudaEmitterFactory()
{
	return new CudaEmitterFactory();
}

} /* namespace partflow */
} /* namespace PFCore */
