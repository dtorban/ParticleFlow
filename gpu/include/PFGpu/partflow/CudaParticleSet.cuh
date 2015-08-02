/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef CUDAPARTICLESET_H_
#define CUDAPARTICLESET_H_

#include "PFCore/env_cuda.h"
#include "PFCore/partflow/ParticleSet.h"

namespace PFCore {
namespace partflow {

class CudaParticleSet : public ParticleSet {
public:
	CudaParticleSet(int numParticles, int numValues, int numVectors, int numSteps = 1);
	CudaParticleSet(int _deviceId, int numParticles, int numValues, int numVectors, int numSteps = 1);
	virtual ~CudaParticleSet();

protected:
	void copy(const ParticleSetView& particleSet, void* dst, const void* src, size_t size);
};

} /* namespace partflow */
}

#endif /* CUDAPARTICLESET_H_ */
