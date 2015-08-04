/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef CUDAVECTORFIELDADVECTOR_H_
#define CUDAVECTORFIELDADVECTOR_H_

#include "PFCore/env_cuda.h"
#include "PFCore/partflow/Advector.h"
#include <string>

namespace PFCore {
namespace partflow {

template<typename Strategy, typename VField>
class CudaVectorFieldAdvector : public Advector {
public:
	CudaVectorFieldAdvector(void* strategy, void* vectorField);
	virtual ~CudaVectorFieldAdvector();
	
	void advectParticles(ParticleSetView& particleSet, int step, float time, float dt);
	
private:
	Strategy _strategy;
	VField _vectorField;
};

template<typename Strategy, typename VField>
inline CudaVectorFieldAdvector<Strategy, VField>::CudaVectorFieldAdvector(void* strategy, void* vectorField) {
	_strategy = *(dynamic_cast<Strategy*>(strategy));
	_vectorField = *(dynamic_cast<VField*>vectorField));
}

template<typename Strategy, typename VField>
inline CudaVectorFieldAdvector<Strategy, VField>::~CudaVectorFieldAdvector() {
}

template<typename Strategy, typename VField>
__global__ void CudaVectorFieldAdvector_advectParticle(Strategy strategy, VField vectorField, ParticleSetView particleSet, int step, float time, float dt)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < particleSet.getNumParticles())
	{
		strategy.advectParticle(particleSet, vectorField, i, step, time, dt);
	}
}


template<typename Strategy, typename VField>
void CudaVectorFieldAdvector::advectParticles(ParticleSetView& particleSet, int step, float time, float dt) {
	CudaVectorFieldAdvector_advectParticle<Strategy, VField><<<particleSet.getNumParticles(), particleSet.getNumParticles()>>>(_strategy, _vectorField, particleSet, step, time, dt);
}

} /* namespace partflow */
} /* namespace PFCore */

#endif /* CUDAVECTORFIELDADVECTOR_H_ */
