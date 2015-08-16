/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef EULERADVECTOR_H_
#define EULERADVECTOR_H_

#include "PFCore/partflow/Advector.h"
#include <string>

namespace PFCore {
namespace partflow {

template<typename VField>
class EulerAdvector {
public:
	PF_ENV_API EulerAdvector() {}
	PF_ENV_API ~EulerAdvector() {}

	PF_ENV_API void advectParticle(ParticleSetView& particleSet, VField vectorField, int index, int step, int prevStep, float time, float dt, int iterations);
	std::string getTypeId() { return "Euler"; }
};

template<typename VField>
PF_ENV_API inline void EulerAdvector<VField>::advectParticle(ParticleSetView& particleSet, VField vectorField, int index, int step, int prevStep, float time, float dt, int iterations)
{
	math::vec3& partPos = particleSet.getPosition(index, step);
	partPos.x = particleSet.getPosition(index, prevStep).x;
	partPos.y = particleSet.getPosition(index, prevStep).y;
	partPos.z = particleSet.getPosition(index, prevStep).z;

	float advectTime = time;

	for (int f = 0; f < iterations; f++)
	{
		math::vec3 velocity = vectorField.getVelocity(partPos, advectTime);
		partPos += velocity*dt;

		if (particleSet.getNumVectors() > 0)
		{
			math::vec3& vel = particleSet.getVector(0, index, step);
			vel = velocity;

		}

		advectTime += dt;
	}
}

} /* namespace partflow */
} /* namespace PFCore */

#endif /* EULERADVECTOR_H_ */
