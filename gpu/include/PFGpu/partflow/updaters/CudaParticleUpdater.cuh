/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef CUDAPARTICLEUPDATER_H_
#define CUDAPARTICLEUPDATER_H_

#include "PFCore/env_cuda.h"
#include "PFCore/partflow/ParticleUpdater.h"
#include <iostream>

namespace PFCore {
namespace partflow {

template<typename Strategy>
class CudaParticleUpdater : public ParticleUpdater {
public:
	CudaParticleUpdater(void* strategy);
	virtual ~CudaParticleUpdater() {}
	
	void updateParticles(ParticleSetView& particleSet, int step, float time);
	
private:
	Strategy _strategy;
};

template<typename Strategy>
CudaParticleUpdater<Strategy>::CudaParticleUpdater(void* strategy) : ParticleUpdater()
{
	_strategy = *(reinterpret_cast<Strategy*>(strategy));
}

template<typename Strategy>
__global__ void CudaParticleUpdater_updateParticles(Strategy strategy, ParticleSetView particleSet, int step, float time)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < particleSet.getNumParticles())
	{
		strategy.updateParticle(particleSet, i, step, time);
	}
}

 template<typename Strategy>
void CudaParticleUpdater<Strategy>::updateParticles(ParticleSetView& particleSet, int step, float time)
{
//	std::cout << "update cuda!" << std::endl;
	CudaParticleUpdater_updateParticles<Strategy><<<1024, particleSet.getNumParticles()/1024>>>(_strategy, particleSet, step, time);
	cudaDeviceSynchronize();
}

} /* namespace partflow */
} /* namespace PFCore */

#endif /* CUDAPARTICLEUPDATER_H_ */
