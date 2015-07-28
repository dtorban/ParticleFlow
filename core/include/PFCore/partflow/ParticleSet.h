/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef PARTICLESET_H_
#define PARTICLESET_H_

#include "PFCore/partflow/ParticleSetView.h"

namespace PFCore {
namespace partflow {

class ParticleSet : public ParticleSetView {
public:
	ParticleSet(int numParticles, int numValues, int numVectors, int numSteps = 1) : ParticleSetView()
	{
		_numParticles = numParticles;
		_numValues = numValues;
		_numVectors = numVectors;
		_numSteps = numSteps;
		_createdArrays = true;
		_positions = new math::vec3[numSteps*numParticles];
		_values = new float[numSteps*numParticles*numValues];
		_vectors = new math::vec3[numSteps*numParticles*numVectors];
	}

	virtual ~ParticleSet()
	{
		if (_createdArrays)
		{
			delete[] _positions;
			delete[] _values;
			delete[] _vectors;
		}
	}

private:
	bool _createdArrays;

protected:
	ParticleSet() : ParticleSetView(), _createdArrays(false)
	{
	}
};

} /* namespace partflow */
} /* namespace PFCore */

#endif /* PARTICLESET_H_ */
