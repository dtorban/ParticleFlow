/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef VECTORFIELDADVECTOR_H_
#define VECTORFIELDADVECTOR_H_

#include "PFCore/partflow/Advector.h"

namespace PFCore {
namespace partflow {

template<typename Strategy, typename VField>
class VectorFieldAdvector : public Advector {
public:
	VectorFieldAdvector(const Strategy& strategy, const VField& vectorField) : _strategy(strategy), _vectorField(vectorField) {}
	virtual ~VectorFieldAdvector() {}

	virtual void advectParticles(ParticleSetView& particleSet, int step, float time, float dt);

	void setVectorField(VField vectorField) {
		_vectorField = vectorField;
	}

private:
	Strategy _strategy;
	VField _vectorField;
};

template<typename Strategy, typename VField>
void VectorFieldAdvector<Strategy, VField>::advectParticles(ParticleSetView& particleSet, int step, float time, float dt)
{
	int prevStep = (particleSet.getNumSteps() + step - 1) % particleSet.getNumSteps();

	for (int index = 0; index < particleSet.getNumParticles(); index++)
	{
		_strategy.advectParticle(particleSet, _vectorField, index, step, prevStep, time, dt);
	}

}

} /* namespace partflow */
} /* namespace PFCore */

#endif /* VECTORFIELDADVECTOR_H_ */
