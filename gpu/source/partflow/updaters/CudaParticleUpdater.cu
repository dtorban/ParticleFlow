/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <PFGpu/partflow/updaters/CudaParticleUpdater.cuh>
#include "PFCore/partflow/updaters/strategies/MagnitudeUpdater.h"

namespace PFCore {
namespace partflow {

ParticleUpdater* createCudaParticleUpdater(std::string strategyTypeId, void* strategy)
{
	if (strategyTypeId == "MagnitudeUpdater")
	{
		return new CudaParticleUpdater<MagnitudeUpdater>(strategy);
	}

	return NULL;
}

} /* namespace partflow */
} /* namespace PFCore */
