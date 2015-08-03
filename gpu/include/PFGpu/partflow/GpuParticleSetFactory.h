/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef CUDAPARTICLESETFACTORY_H_
#define CUDAPARTICLESETFACTORY_H_

#include "PFCore/partflow/ParticleSetFactory.h"

#include <iostream>

namespace PFCore {
namespace partflow {

#ifdef USE_CUDA
extern "C"
ParticleSet* createCudaParticleSet(int deviceId, int numParticles, int numValues, int numVectors, int numSteps);
#endif

class GpuParticleSetFactory : public ParticleSetFactory {
public:
	GpuParticleSetFactory() : ParticleSetFactory() {}
	virtual ~GpuParticleSetFactory() {}

	ParticleSetRef createParticleSet(int deviceId, int numParticles, int numValues = 0, int numVectors = 0, int numSteps = 1)
	{
#ifdef USE_CUDA
		std::cout << "Use cuda particle set" << std::endl;
		return ParticleSetRef(createCudaParticleSet(deviceId, numParticles, numValues, numVectors, numSteps));
#else
		std::cout << "Use cpu particle set" << std::endl;
		return ParticleSetFactory::createParticleSet(deviceId, numParticles, numValues, numVectors, numSteps);
#endif
	}
};

} /* namespace partflow */
} /* namespace PFCore */

#endif /* CUDAPARTICLESETFACTORY_H_ */
