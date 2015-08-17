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
#include "PFGpu/GpuResource.h"

namespace PFCore {
namespace partflow {

class CudaParticleSet : public ParticleSet {
public:
	CudaParticleSet(int numParticles, int numAttributes, int numValues, int numVectors, int numSteps);
	CudaParticleSet(int _deviceId, int numParticles, int numAttributes, int numValues, int numVectors, int numSteps);
	CudaParticleSet(GpuResource* resource, int numParticles, int numAttributes, int numValues, int numVectors, int numSteps);
	virtual ~CudaParticleSet();

protected:
	void copy(const ParticleSetView& particleSet, void* dst, const void* src, size_t size);
	
private:
	bool _createdArrays;
};

} /* namespace partflow */
}

#endif /* CUDAPARTICLESET_H_ */
