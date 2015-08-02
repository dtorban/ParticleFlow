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

CudaParticleSet::CudaParticleSet(int numParticles, int numValues, int numVectors, int numSteps) : ParticleSet(numParticles, numValues, numVectors, numSteps), _deviceId(-1)
{
}

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
	if (_deviceId >= 0)
	{
		cudaSetDevice(_deviceId);
		cudaFree(_positions);
		cudaFree(_values);
		cudaFree(_vectors);
	}
}

void CudaParticleSet::copy(const ParticleSetView& particleSet, void* dst, const void* src, size_t size)
{
	// Both local
	if (getDeviceId() < 0 && particleSet.getDeviceId() < 0)
	{
		ParticleSet::copy(particleSet, dst, src, size);
	}
	// On same device
	else if (getDeviceId() == particleSet.getDeviceId())
	{
		// TODO: kernal to copy to same device
		cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
	}
	// One of the devices is host
	else if (getDeviceId() < 0 || particleSet.getDeviceId() < 0)
	{
		cudaMemcpy(dst, src, size, getDeviceId() < 0 ? cudaMemcpyDeviceToHost :  cudaMemcpyHostToDevice);
	}
	// Peer to peer
	else
	{
		cudaMemcpyPeer(dst, getDeviceId(), src, particleSet.getDeviceId(), size);
	}
}

extern "C"
ParticleSet* createCudaParticleSet(int deviceId, int numParticles, int numValues, int numVectors, int numSteps)
{
	if (deviceId >= 0)
	{
		return new CudaParticleSet(deviceId, numParticles, numValues, numVectors, numSteps);
	}
	else
	{
		return new CudaParticleSet(numParticles, numValues, numVectors, numSteps);
	}
}

} /* namespace partflow */
}
