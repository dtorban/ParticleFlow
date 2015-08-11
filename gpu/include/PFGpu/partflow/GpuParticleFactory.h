/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef CUDAPARTICLESETFACTORY_H_
#define CUDAPARTICLESETFACTORY_H_

#include "PFCore/partflow/ParticleFactory.h"

#include <iostream>

namespace PFCore {
namespace partflow {

#ifdef USE_CUDA
extern "C"
ParticleSet* createCudaParticleSet(int deviceId, int numParticles, int numValues, int numVectors, int numSteps);
#endif

#ifdef USE_CUDA
extern "C"
ParticleField* createCudaParticleField(int deviceId, int numValues, int numVectors, math::vec4 start, math::vec4 length, math::vec4 size);
#endif

class GpuParticleFactory : public ParticleFactory {
public:
	GpuParticleFactory() : ParticleFactory() {}
	virtual ~GpuParticleFactory() {}

	ParticleSetRef createParticleSet(int deviceId, int numParticles, int numValues = 0, int numVectors = 1, int numSteps = 1)
	{
#ifdef USE_CUDA
		std::cout << "Use cuda particle set" << std::endl;
		return ParticleSetRef(createCudaParticleSet(deviceId, numParticles, numValues, numVectors, numSteps));
#else
		std::cout << "Use cpu particle set" << std::endl;
		return ParticleFactory::createParticleSet(deviceId, numParticles, numValues, numVectors, numSteps);
#endif
	}

	ParticleFieldRef createParticleField(int deviceId, int numValues, int numVectors, math::vec4 start, math::vec4 length, math::vec4 size)
	{
#ifdef USE_CUDA
		std::cout << "Use cuda particle set" << std::endl;
		return ParticleFieldRef(createCudaParticleField(deviceId, numValues, numVectors, start, length, size));
#else
		std::cout << "Use cpu particle set" << std::endl;
		return ParticleFactory::createParticleField(deviceId, numValues, numVectors, start, length, size);
#endif
	}
};

} /* namespace partflow */
} /* namespace PFCore */

#endif /* CUDAPARTICLESETFACTORY_H_ */