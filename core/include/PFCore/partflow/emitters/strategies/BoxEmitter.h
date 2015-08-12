/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef BOXEMITTER_H_
#define BOXEMITTER_H_

#include "PFCore/partflow/Emitter.h"
#include "PFCore/math/v3.h"
#include "PFCore/math/RandomValue.h"
#include "PFCore/math/vec_math.h"

namespace PFCore {
namespace partflow {

class BoxEmitter {
public:
	BoxEmitter(const math::vec3 &low, const math::vec3 &high, int duration) : _low(low), _length(high-low), _duration(duration) {}
	PF_ENV_API BoxEmitter(const BoxEmitter& emitter) {
		_low = emitter._low;
		_length = emitter._length;
		_duration = emitter._duration;
	}
	PF_ENV_API ~BoxEmitter() {}

	PF_ENV_API inline void emitParticle(ParticleSetView& particleSet, int index, int step, math::RandomValue rnd, bool init);

private:
	math::vec3 _low;
	math::vec3 _length;
	int _duration;
};

PF_ENV_API inline void BoxEmitter::emitParticle(ParticleSetView& particleSet, int index, int step, math::RandomValue rnd, bool init) {

	if (init || (index + step) % _duration == 0)
	{
		math::vec3& partPos = particleSet.getPosition(index, step);
		partPos = _low;
		partPos.x += _length.x*rnd.getValue(index);
		partPos.y += _length.y*rnd.getValue(index+1);
		partPos.z += _length.z*rnd.getValue(index+2);
	}
}

} /* namespace partflow */
} /* namespace PFCore */

#endif /* BOXEMITTER_H_ */
