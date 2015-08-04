/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef MAGNITUDEUPDATER_H_
#define MAGNITUDEUPDATER_H_

#include "PFCore/partflow/ParticleUpdater.h"
#include "PFCore/math/vec_math.h"

namespace PFCore {
namespace partflow {

class MagnitudeUpdater {
public:
	PF_ENV_API MagnitudeUpdater(const MagnitudeUpdater& updater) { (*this) = updater; }
	PF_ENV_API MagnitudeUpdater() : _valueIndex(0), _vectorIndex(0) {}
	PF_ENV_API MagnitudeUpdater(int valueIndex, int vectorIndex) : _valueIndex(valueIndex), _vectorIndex(vectorIndex) {}
	PF_ENV_API ~MagnitudeUpdater() {}
	PF_ENV_API inline void operator=(const MagnitudeUpdater& updater);

	PF_ENV_API inline void updateParticle(ParticleSetView& particleSet, int index, int step);

private:
	int _valueIndex;
	int _vectorIndex;
};

PF_ENV_API inline void MagnitudeUpdater::operator=(const MagnitudeUpdater& updater)
{
	_valueIndex = updater._valueIndex;
	_vectorIndex = updater._vectorIndex;
}

PF_ENV_API inline void MagnitudeUpdater::updateParticle(ParticleSetView& particleSet, int index, int step)
{
	particleSet.getValue(_valueIndex, index, step) = math::length(particleSet.getVector(_vectorIndex, index, step));
}

} /* namespace partflow */
} /* namespace PFCore */

#endif /* MAGNITUDEUPDATER_H_ */
