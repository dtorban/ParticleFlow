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
#include "PFGpu/math/CudaRandomValue.cuh"
#include <map>
#include <iostream>
#include <stdio.h>

namespace PFCore {
namespace partflow {

template<typename Strategy>
class CudaEmitter : public BasicEmitter<Strategy> {
public:
	CudaEmitter(const Strategy& strategy);
	virtual ~CudaEmitter();
	
	void emitParticles(ParticleSetView& particleSet, int step, bool init);

private:
	std::map<int, math::CudaRandomValue*> _randValues;
};

template<typename Strategy>
inline CudaEmitter<Strategy>::CudaEmitter(const Strategy& strategy) : BasicEmitter<Strategy>(strategy), _randValues()
{
	std::cout << "Created strategry." << std::endl;
}

template<typename Strategy>
inline CudaEmitter<Strategy>::~CudaEmitter()
{
	for (std::map<int, math::CudaRandomValue*>::iterator it=_randValues.begin(); it!=_randValues.end(); ++it)
    {
    	delete it->second;
    }
}

template<typename Strategy>
__global__ void CudaEmitter_emitParticle(Strategy strategy, ParticleSetView particleSet, int step, math::RandomValue rnd, bool init)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < particleSet.getNumParticles())
	{
		strategy.emitParticle(particleSet, i, step, rnd, init);
	}
}

template<typename Strategy>
inline void CudaEmitter<Strategy>::emitParticles(ParticleSetView& particleSet, int step, bool init) {
	int deviceId = particleSet.getDeviceId();
	if (deviceId < 0)
	{
		BasicEmitter<Strategy>::emitParticles(particleSet, step, init);
		return;
	}

	cudaSetDevice(deviceId);
	if (_randValues.find(deviceId) == _randValues.end())
	{
		_randValues[deviceId] = new math::CudaRandomValue(deviceId, 1024*1024);
	}
	
	math::RandomValue rnd = *(_randValues[deviceId]);
	rnd.randomize(0);
	
	//std::cout << "Cuda emit." << std::endl;
	CudaEmitter_emitParticle<Strategy><<<1024, particleSet.getNumParticles()/1024>>>(this->_strategy, particleSet, step, rnd, init);
	cudaDeviceSynchronize();
}

} /* namespace partflow */
}

#endif /* CUDAEMITTER_H_ */
