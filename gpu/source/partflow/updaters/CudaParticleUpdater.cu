/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <PFGpu/partflow/updaters/CudaParticleUpdater.cuh>
#include "PFCore/partflow/updaters/strategies/MagnitudeUpdater.h"
#include "PFCore/partflow/updaters/strategies/ParticleFieldUpdater.h"

namespace PFCore {
namespace partflow {

extern "C"
ParticleUpdater* createCudaParticleUpdater(std::string strategyTypeId, void* strategy)
{
	if (strategyTypeId == "MagnitudeUpdater")
	{
		return new CudaParticleUpdater<MagnitudeUpdater>(strategy);
	}
	else if (strategyTypeId == "ParticleFieldUpdater")
	{
		return new CudaParticleUpdater<ParticleFieldUpdater>(strategy);
	}

	return NULL;
}

} /* namespace partflow */
} /* namespace PFCore */
