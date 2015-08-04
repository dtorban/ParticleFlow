/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef GPUPARTICLEUPDATER_H_
#define GPUPARTICLEUPDATER_H_

#include "PFCore/partflow/ParticleUpdater.h"
#include "PFCore/partflow/updaters/BasicUpdater.h"

namespace PFCore {
namespace partflow {

#ifdef USE_CUDA
extern "C"
ParticleUpdater* createCudaParticleUpdater(std::string strategyTypeId, void* strategy);
#endif

template<typename Strategy>
class GpuParticleUpdater : public ParticleUpdater {
public:
	GpuParticleUpdater(Strategy strategy);
	virtual ~GpuParticleUpdater();

	void updateParticles(ParticleSetView& particleSet, int step);

private:
	BasicUpdater<Strategy> _localUpdater;
#ifdef USE_CUDA
	ParticleUpdater* _innerUpdater;
#endif
};

template<typename Strategy>
inline GpuParticleUpdater<Strategy>::GpuParticleUpdater(Strategy strategy) : _localUpdater(strategy)
{
#ifdef USE_CUDA
	_innerUpdater = createCudaParticleUpdater(strategy.getTypeId(), &strategy);
#endif
}

template<typename Strategy>
inline GpuParticleUpdater<Strategy>::~GpuParticleUpdater() {
#ifdef USE_CUDA
	delete _innerUpdater;
#endif
}

template<typename Strategy>
inline void GpuParticleUpdater<Strategy>::updateParticles(ParticleSetView& particleSet, int step)
{
#ifdef USE_CUDA
	if (particleSet.getDeviceId() >= 0)
	{
		_innerUpdater->updateParticles(particleSet, step);
		return;
	}
#endif

	_localUpdater.updateParticles(particleSet, step);
}

} /* namespace partflow */
} /* namespace PFCore */

#endif /* GPUPARTICLEUPDATER_H_ */
