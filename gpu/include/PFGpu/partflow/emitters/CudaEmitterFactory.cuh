/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */
 
 #include "PFCore/env_cuda.h"
 #include "PFCore/partflow/emitters/EmitterFactory.h"

#ifndef CUDAEMITTERFACTORY_H_
#define CUDAEMITTERFACTORY_H_

#include "PFGpu/math/CudaRandomValue.cuh"

namespace PFCore {
namespace partflow {

class CudaEmitterFactory : public EmitterFactory {
public:
	CudaEmitterFactory();
	virtual ~CudaEmitterFactory();
	
	Emitter* createSphereEmitter(math::vec3 pos, float radius, int duration);
};

} /* namespace partflow */
} /* namespace PFCore */

#endif /* CUDAEMITTERFACTORY_H_ */
