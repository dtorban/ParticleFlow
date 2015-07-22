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
	CudaParticleSet(int _deviceId, int numParticles, int numValues, int numVectors);
	virtual ~CudaParticleSet();

private:
	int _deviceId;
};

} /* namespace partflow */
}

#endif /* CUDAPARTICLESET_H_ */