/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef RUNGAKUTTA4_H_
#define RUNGAKUTTA4_H_

#include "PFCore/partflow/Advector.h"
#include <string>

namespace PFCore {
namespace partflow {

template<typename VField>
class RungaKutta4 {
public:
	PF_ENV_API RungaKutta4() {}
	PF_ENV_API ~RungaKutta4() {}

	PF_ENV_API void advectParticle(ParticleSetView& particleSet, VField vectorField, int index, int step, int prevStep, float time, float dt, int iterations);
	std::string getTypeId() { return "RungaKutta4"; }
};

template<typename VField>
PF_ENV_API inline void RungaKutta4<VField>::advectParticle(ParticleSetView& particleSet, VField vectorField, int index, int step, int prevStep, float time, float dt, int iterations)
{
	math::vec3& partPos = particleSet.getPosition(index, step);
	partPos.x = particleSet.getPosition(index, prevStep).x;
	partPos.y = particleSet.getPosition(index, prevStep).y;
	partPos.z = particleSet.getPosition(index, prevStep).z;

	math::vec3 k1, k2, k3, k4, v, a;
	float advectTime = time;

	for (int f = 0; f < iterations; f++)
	{
		k1 = vectorField.getVelocity(partPos, advectTime);
		k2 = vectorField.getVelocity(partPos + (k1 * 0.5), advectTime+0.5*dt);
		k3 = vectorField.getVelocity(partPos + (k2 * 0.5), advectTime+0.5*dt);
		k4 = vectorField.getVelocity(partPos + k3, advectTime);

		v = (k1/6.0) + (k2/3.0) + (k3/3.0) + (k4/6.0);

		partPos += v*dt;

		if (particleSet.getNumVectors() > 0)
		{
			particleSet.getVector(0, index, step) = v;
		}

		advectTime+=dt;
	}
}

} /* namespace partflow */
} /* namespace PFCore */

#endif /* RUNGAKUTTA4_H_ */
