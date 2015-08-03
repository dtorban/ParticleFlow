/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef SPHEREEMITTER_H_
#define SPHEREEMITTER_H_

#include "PFCore/partflow/Emitter.h"
#include "PFCore/math/v3.h"
#include "PFCore/math/RandomValue.h"
#include "PFCore/math/vec_math.h"

namespace PFCore {
namespace partflow {

class SphereEmitter {
public:
	SphereEmitter(math::vec3 pos, float radius, int duration) : _pos(pos), _radius(radius), _duration(duration) {}
	PF_ENV_API SphereEmitter(const SphereEmitter& emitter) {
		_pos = emitter._pos;
		_radius = emitter._radius;
		_duration = emitter._duration;
	}
	PF_ENV_API ~SphereEmitter() {}

	PF_ENV_API inline void preEmit();
	PF_ENV_API inline void emitParticle(ParticleSetView& particleSet, int index, int step, math::RandomValue rnd, bool init);

private:
	math::vec3 _pos;
	float _radius;
	int _duration;
};

PF_ENV_API inline void SphereEmitter::emitParticle(ParticleSetView& particleSet, int index, int step, math::RandomValue rnd, bool init) {

	if (init || index % _duration == 0)
	{
		math::vec3& partPos = particleSet.getPosition(index, step);
		partPos.x = rnd.getValue(index)*2.0-1.0;
		partPos.y = rnd.getValue(index+1)*2.0-1.0;
		partPos.z = rnd.getValue(index+2)*2.0-1.0;
		float len = math::length(partPos);
		if (len > 0.0f)
		{
			partPos /= len;
		}
		partPos *= rnd.getValue(index+4)*_radius;
		partPos += _pos;
	}
}

} /* namespace partflow */
} /* namespace PFCore */


#endif /* SPHEREEMITTER_H_ */
