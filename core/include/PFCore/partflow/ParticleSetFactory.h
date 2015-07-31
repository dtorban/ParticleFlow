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
#include <memory>

namespace PFCore {
namespace partflow {

typedef std::shared_ptr<ParticleSet> ParticleSetRef;

class ParticleSetFactory {
public:
	ParticleSetFactory() {}
	virtual ~ParticleSetFactory() {}

	ParticleSetRef createParticleSet(int numParticles, int numValues = 0, int numVectors = 0, int numSteps = 1)
	{
		return createParticleSet(-1, numParticles, numValues, numVectors, numSteps);
	}

	virtual ParticleSetRef createParticleSet(int deviceId, int numParticles, int numValues = 0, int numVectors = 0, int numSteps = 1)
	{
		return ParticleSetRef(new ParticleSet(numParticles, numValues, numVectors, numSteps));
	}
};

} /* namespace partflow */
} /* namespace PFCore */

#endif /* PARTICLESETFACTORY_H_ */
