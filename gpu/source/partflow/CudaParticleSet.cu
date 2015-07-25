/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <gpu/include/PFGpu/partflow/CudaParticleSet.cuh>

namespace PFCore {
namespace partflow {

CudaParticleSet::CudaParticleSet(int deviceId, int numParticles, int numValues, int numVectors, int numSteps) : ParticleSet(), _deviceId(deviceId) {
	_numParticles = numParticles;
	_numValues = numValues;
	_numVectors = numVectors;
	_numSteps = numSteps;
	cudaSetDevice(deviceId);
	cudaMalloc(&_positions, numSteps*numParticles*sizeof(math::vec3));
	cudaMalloc(&_values, numSteps*numParticles*numValues*sizeof(float));
	cudaMalloc(&_vectors, numSteps*numParticles*numVectors*sizeof(math::vec3));
}

CudaParticleSet::~CudaParticleSet() {
	cudaSetDevice(_deviceId);
	cudaFree(_positions);
	cudaFree(_values);
	cudaFree(_vectors);
}

ParticleSet* createCudaParticleSet(int deviceId, int numParticles, int numValues, int numVectors, int numSteps)
{
	return new CudaParticleSet(deviceId, numParticles, numValues, numVectors, numSteps);
}

} /* namespace partflow */
}
