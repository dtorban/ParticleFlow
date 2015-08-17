/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <gpu/include/PFGpu/partflow/CudaParticleSet.cuh>
#include "gpu/include/PFGpu/CudaHelper.cuh"

namespace PFCore {
namespace partflow {

CudaParticleSet::CudaParticleSet(int numParticles, int numAttributes, int numValues, int numVectors, int numSteps) : ParticleSet(numParticles, numAttributes, numValues, numVectors, numSteps)
{
	_deviceId = -1;
	_createdArrays = false;
}

CudaParticleSet::CudaParticleSet(int deviceId, int numParticles, int numAttributes, int numValues, int numVectors, int numSteps) : ParticleSet() {
	_numParticles = numParticles;
	_numAttributes = _numAttributes;
	_numValues = numValues;
	_numVectors = numVectors;
	_numSteps = numSteps;
	_deviceId = deviceId;
	_createdArrays = true;
	cudaSetDevice(deviceId);
	cudaMalloc(&_positions, numSteps*numParticles*sizeof(math::vec3));
	cudaMalloc(&_attributes, numSteps*numParticles*numAttributes*sizeof(int));
	cudaMalloc(&_values, numSteps*numParticles*numValues*sizeof(float));
	cudaMalloc(&_vectors, numSteps*numParticles*numVectors*sizeof(math::vec3));
}

CudaParticleSet::CudaParticleSet(GpuResource* resource, int numParticles, int numAttributes, int numValues, int numVectors, int numSteps) : ParticleSet() {
	_numParticles = numParticles;
	_numAttributes = _numAttributes;
	_numValues = numValues;
	_numVectors = numVectors;
	_numSteps = numSteps;
	_deviceId = resource->getDeviceId();
	_createdArrays = false;
	cudaSetDevice(_deviceId);
	
	char *data;
	int size = resource->getData((void **)(&data));
	int numInstances = numParticles*numSteps;
	int startPos = 0;
	_positions = reinterpret_cast<PFCore::math::vec3*>(&data[startPos]);
	startPos += sizeof(PFCore::math::vec3)*numInstances;
	_attributes = reinterpret_cast<int*>(&data[startPos]);
	startPos += sizeof(int)*numInstances*numAttributes;
	_values = reinterpret_cast<float*>(&data[startPos]);
	startPos += sizeof(float)*numInstances*numValues;
	_vectors = reinterpret_cast<PFCore::math::vec3*>(&data[startPos]);
}

CudaParticleSet::~CudaParticleSet() {
	if (_createdArrays)
	{
		cudaSetDevice(_deviceId);
		cudaFree(_positions);
		cudaFree(_attributes);
		cudaFree(_values);
		cudaFree(_vectors);
	}
}

void CudaParticleSet::copy(const ParticleSetView& particleSet, void* dst, const void* src, size_t size)
{
	CudaHelper::copy(dst, getDeviceId(), src, particleSet.getDeviceId(), size);
}

extern "C"
ParticleSet* createCudaParticleSet(int deviceId, int numParticles, int numAttributes, int numValues, int numVectors, int numSteps)
{
	if (deviceId >= 0)
	{
		return new CudaParticleSet(deviceId, numParticles, numAttributes, numValues, numVectors, numSteps);
	}
	else
	{
		return new CudaParticleSet(numParticles, numAttributes, numValues, numVectors, numSteps);
	}
}

extern "C"
ParticleSet* createCudaParticleSetFromResource(GpuResource* resource, int numParticles, int numAttributes, int numValues, int numVectors, int numSteps)
{
	if (resource->getDeviceId() >= 0)
	{
		return new CudaParticleSet(resource, numParticles, numAttributes, numValues, numVectors, numSteps);
	}
	else
	{
		return new CudaParticleSet(numParticles, numAttributes, numValues, numVectors, numSteps);
	}
}

} /* namespace partflow */
}
