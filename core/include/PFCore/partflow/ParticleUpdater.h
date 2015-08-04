/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef PARTICLEUPDATER_H_
#define PARTICLEUPDATER_H_

#include "PFCore/partflow/ParticleSetView.h"

namespace PFCore {
namespace partflow {

class ParticleUpdater {
public:
	virtual ~ParticleUpdater() {}

	virtual void updateParticles(ParticleSetView& particleSet, int step) = 0;
};

} /* namespace partflow */
} /* namespace PFCore */

#endif /* PARTICLEUPDATER_H_ */
