/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef CUDAPARTICLEUPDATER_H_
#define CUDAPARTICLEUPDATER_H_

#include "PFCore/partflow/ParticleUpdater.h"

namespace PFCore {
namespace partflow {

template<typename Strategy>
class CudaParticleUpdater : public ParticleUpdater {
public:
	CudaParticleUpdater(void* strategy);
	virtual ~CudaParticleUpdater() {}
	
	void updateParticles(ParticleSetView& particleSet, int step);
	
private:
	Strategy _strategy;
};

template<typename Strategy>
CudaParticleUpdater<Strategy>::CudaParticleUpdater(void* strategy)
{
	_strategy = *(reinterpret_cast<Strategy*>(strategy));
}

template<typename Strategy>
__global__ void CudaParticleUpdater_updateParticles(Strategy strategy, ParticleSetView particleSet, int step)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < particleSet.getNumParticles())
	{
		strategy.updateParticle(particleSet, i, step);
	}
}

 template<typename Strategy>
void CudaParticleUpdater<Strategy>::updateParticles(ParticleSetView& particleSet, int step)
{
	std::cout << "update cuda!" << std::endl;
	CudaParticleUpdater_updateParticles<Strategy><<<particleSet.getNumParticles(), particleSet.getNumParticles()>>>(_strategy, particleSet, step);
}

} /* namespace partflow */
} /* namespace PFCore */

#endif /* CUDAPARTICLEUPDATER_H_ */
