/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef CUDAEMITTER_H_
#define CUDAEMITTER_H_

#include "PFCore/env_cuda.h"
#include "PFCore/partflow/emitters/BasicEmitter.h"

namespace PFCore {
namespace partflow {

template<typename Strategy>
class CudaEmitter : public BasicEmitter<Strategy> {
public:
	CudaEmitter(int deviceId, const Strategy& strategy);
	virtual ~CudaEmitter();
	
	void emitParticles(ParticleSetView& particleSet, int step, bool init);

private:
	int _deviceId;
};

template<typename Strategy>
inline CudaEmitter<Strategy>::CudaEmitter(int deviceId, const Strategy& strategy) : BasicEmitter(strategy), _deviceId(deviceId)
{
}

template<typename Strategy>
inline CudaEmitter<Strategy>::~CudaEmitter()
{
}

template<typename Strategy>
__global__ void CudaEmitter_emitParticle(Strategy strategy, ParticleSetView particleSet, int step, bool init)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < particleSet.getNumParticles())
	{
		strategy.emitParticle(particleSet, i, step, init);
	}
}

template<typename Strategy>
inline void CudaEmitter<Strategy>::emitParticles(ParticleSetView& particleSet, int step, bool init) {

	cudaSetDevice(_deviceId);
	CudaEmitter_emitParticle<Strategy><<<1024, particleSet.getNumParticles()/1024>>>(_strategy, particleSet, step, init);
}

} /* namespace partflow */
}

#endif /* CUDAEMITTER_H_ */
