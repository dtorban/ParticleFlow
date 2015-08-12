/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef PARTICLEFIELDUPDATER_H_
#define PARTICLEFIELDUPDATER_H_

#include "PFCore/partflow/ParticleUpdater.h"
#include "PFCore/partflow/vectorFields/ParticleFieldVolume.h"
#include <iostream>

namespace PFCore {
namespace partflow {

class ParticleFieldUpdater {
public:
	//PF_ENV_API ParticleFieldUpdater() {}
	PF_ENV_API ParticleFieldUpdater(const ParticleFieldUpdater& updater) { (*this) = updater; }
	PF_ENV_API ParticleFieldUpdater(const ParticleFieldVolume& volume) : _volume(volume) {}
	PF_ENV_API ~ParticleFieldUpdater() {}
	PF_ENV_API inline void operator=(const ParticleFieldUpdater& updater);

	PF_ENV_API inline void updateParticle(ParticleSetView& particleSet, int index, int step);

	std::string getTypeId() { return "ParticleFieldUpdater"; }

private:
	ParticleFieldVolume _volume;
};

PF_ENV_API inline void ParticleFieldUpdater::operator=(const ParticleFieldUpdater& updater)
{
	_volume = updater._volume;
}

PF_ENV_API inline void ParticleFieldUpdater::updateParticle(ParticleSetView& particleSet, int index, int step)
{
	int numValues = _volume.getParticleField().getNumValues() < particleSet.getNumValues() ? _volume.getParticleField().getNumValues() : particleSet.getNumValues();
	int numVectors = _volume.getParticleField().getNumVectors() < particleSet.getNumVectors() ? _volume.getParticleField().getNumVectors() : particleSet.getNumVectors();

	math::vec3& pos = particleSet.getPosition(index, step);
	math::vec4 inc = _volume.getParticleField().getIncrement();
	float time = inc.t * (step % (int)(_volume.getParticleField().getSize().t));


	for (int f = 0; f < numValues; f++)
	{
		particleSet.getValue(f, index, step) = _volume.getValue(f, pos, time);
	}

	for (int f = 0; f < numVectors; f++)
	{
		particleSet.getVector(f, index, step) = _volume.getVector(f, pos, time);
	}
}

} /* namespace partflow */
} /* namespace PFCore */

#endif /* PARTICLEFIELDUPDATER_H_ */
