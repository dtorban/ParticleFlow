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
	SphereEmitter(math::vec3 pos, float radius, int duration);
	virtual ~SphereEmitter();

	void emitParticles(ParticleSetView& particleSet, int step);

protected:
	virtual void emitParticles(ParticleSetView& particleSet, int step, math::vec3 pos, float radius, int duration);
	PF_ENV_API inline void emitParticle(int index, ParticleSetView& particleSet, int step, const math::vec3 &pos, float radius, const RandomValue& rnd, int duration);
	PF_ENV_API inline void initParticle(int index, ParticleSetView& particleSet, int step, const math::vec3 &pos, float radius, const RandomValue& rnd, int duration);

private:
	math::vec3 _pos;
	float _radius;
	int _duration;
	RandomValue _rnd;
};


SphereEmitter::SphereEmitter(math::vec3 pos, float radius, int duration) : _pos(pos), _radius(radius), _duration(duration) {
}

SphereEmitter::~SphereEmitter() {
}

void SphereEmitter::emitParticles(ParticleSetView& particleSet, int step) {
	emitParticles(particleSet, step, _pos, _radius, _duration);
}

void SphereEmitter::emitParticles(ParticleSetView& particleSet, int step,
		math::vec3 pos, float radius, int duration) {

	for (int f = 0; f < particleSet.getNumParticles(); f++)
	{
		emitParticle(f, particleSet, step, pos, radius, _rnd, duration);
	}
}

PF_ENV_API inline void SphereEmitter::initParticle(int index, ParticleSetView& particleSet, int step,
		const math::vec3 &pos, float radius, const RandomValue& rnd, int duration) {

	math::vec3& partPos = particleSet.getPosition(index, step);
	partPos.x = rnd.getValue(index)*2.0-1.0;
	partPos.y = rnd.getValue(index+1)*2.0-1.0;
	partPos.z = rnd.getValue(index+2)*2.0-1.0;
	float len = math::length(partPos);
	if (len > 0.0f)
	{
		partPos /= len;
	}
	partPos *= rnd.getValue(index+4)*radius;
	partPos += pos;
}

PF_ENV_API inline void SphereEmitter::emitParticle(int index, ParticleSetView& particleSet, int step,
		const math::vec3 &pos, float radius, const RandomValue& rnd, int duration) {

	if (index % duration == 0)
	{
		initParticle(index, particleSet, step, pos, radius, rnd, duration);
	}
}

} /* namespace partflow */
} /* namespace PFCore */


#endif /* SPHEREEMITTER_H_ */
