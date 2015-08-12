/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef PARTICLESETFACTORY_H_
#define PARTICLESETFACTORY_H_

#include "PFCore/partflow/ParticleSet.h"
#include "PFCore/partflow/PartflowRef.h"

namespace PFCore {
namespace partflow {


class ParticleFactory {
public:
	ParticleFactory() {}
	virtual ~ParticleFactory() {}

	ParticleSetRef createLocalParticleSet(int numParticles, int numAttributes = 0, int numValues = 0, int numVectors = 0, int numSteps = 1)
	{
		return createParticleSet(-1, numParticles, numAttributes, numValues, numVectors, numSteps);
	}

	virtual ParticleSetRef createParticleSet(int deviceId, int numParticles, int numAttributes = 0, int numValues = 0, int numVectors = 0, int numSteps = 1)
	{
		return ParticleSetRef(new ParticleSet(numParticles, numAttributes, numValues, numVectors, numSteps));
	}

	ParticleFieldRef createLocalParticleField(int numValues, int numVectors, math::vec4 start, math::vec4 length, math::vec4 size)
	{
		return createParticleField(-1, numValues, numVectors, start, length, size);
	}

	virtual ParticleFieldRef createParticleField(int deviceId, int numValues, int numVectors, math::vec4 start, math::vec4 length, math::vec4 size)
	{
		return ParticleFieldRef(new ParticleField(numValues, numVectors, start, length, size));
	}
};

} /* namespace partflow */
} /* namespace PFCore */

#endif /* PARTICLESETFACTORY_H_ */
