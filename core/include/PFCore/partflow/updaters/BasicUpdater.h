/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef BASICUPDATER_H_
#define BASICUPDATER_H_

#include "PFCore/partflow/ParticleUpdater.h"

namespace PFCore {
namespace partflow {

template<typename Strategy>
class BasicUpdater : public ParticleUpdater {
public:
	BasicUpdater(const Strategy& strategy) : _strategy(strategy) {}
	virtual ~BasicUpdater() {}

	void updateParticles(ParticleSetView& particleSet, int step);

private:
	Strategy _strategy;
};

template<typename Strategy>
void BasicUpdater<Strategy>::updateParticles(ParticleSetView& particleSet, int step) {
	for (int index = 0; index < particleSet.getNumParticles(); index++)
	{
		_strategy.updateParticle(particleSet, index, step);
	}
}

} /* namespace partflow */
} /* namespace PFCore */

#endif /* BASICUPDATER_H_ */
