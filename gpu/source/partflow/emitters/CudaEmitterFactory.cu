/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <PFGpu/partflow/emitters/CudaEmitterFactory.cuh>
#include "PFGpu/partflow/emitters/CudaEmitter.cuh"

namespace PFCore {
namespace partflow {

CudaEmitterFactory::CudaEmitterFactory() : EmitterFactory() {
	// TODO Auto-generated constructor stub

}

CudaEmitterFactory::~CudaEmitterFactory() {
	// TODO Auto-generated destructor stub
}

Emitter* CudaEmitterFactory::createSphereEmitter(math::vec3 pos, float radius, int duration)
{
	return new CudaEmitter<SphereEmitter>(SphereEmitter(pos, radius, duration));
}

Emitter* CudaEmitterFactory::createBoxEmitter(const math::vec3 &low, const math::vec3 &high, int duration)
{
	return new CudaEmitter<BoxEmitter>(BoxEmitter(low, high, duration));
}

extern "C"
EmitterFactory* createCudaEmitterFactory()
{
	return new CudaEmitterFactory();
}

} /* namespace partflow */
} /* namespace PFCore */
