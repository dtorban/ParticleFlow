/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef CUDAPARTICLEFIELD_H_
#define CUDAPARTICLEFIELD_H_

#include "PFCore/env_cuda.h"
#include "PFCore/partflow/vectorFields/ParticleField.h"

namespace PFCore {
namespace partflow {

class CudaParticleField : public ParticleField {
public:
	CudaParticleField(int numValues, int numVectors, math::vec4 start, math::vec4 length, math::vec4 size);
	CudaParticleField(int deviceId, int numValues, int numVectors, math::vec4 start, math::vec4 length, math::vec4 size);
	virtual ~CudaParticleField();

protected:
	void copy(const ParticleFieldView& particleField, void* dst, const void* src, size_t size);
};

} /* namespace partflow */
} /* namespace PFCore */

#endif /* CUDAPARTICLEFIELD_H_ */
