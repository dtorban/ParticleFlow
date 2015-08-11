/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <PFGpu/partflow/vectorFields/CudaParticleField.cuh>
#include "PFGpu/CudaHelper.cuh"

namespace PFCore {
namespace partflow {

CudaParticleField::CudaParticleField(int deviceId, int numValues, int numVectors, math::vec4 start, math::vec4 length, math::vec4 size) {
	_deviceId = deviceId;
	_numValues = numValues;
	_numVectors = numVectors;
	_start = start;
	_length = length;
	_size = size;
	cudaSetDevice(_deviceId);
	cudaMalloc(&_values, size.x*size.y*size.z*size.t*_numValues*sizeof(float));
	cudaMalloc(&_vectors, size.x*size.y*size.z*size.t*_numVectors*sizeof(math::vec3));
}

CudaParticleField::CudaParticleField(int numValues, int numVectors, math::vec4 start, math::vec4 length, math::vec4 size) : ParticleField(numValues, numVectors, start, length, size) {

	_deviceId = -1;
}

CudaParticleField::~CudaParticleField()
{
	if (_deviceId >= 0)
	{
		cudaSetDevice(_deviceId);
		cudaFree(_values);
		cudaFree(_vectors);
	}
}

void CudaParticleField::copy(const ParticleFieldView& particleField, void* dst, const void* src, size_t size)
{
	CudaHelper::copy(dst, getDeviceId(), src, particleField.getDeviceId(), size);
}

extern "C"
ParticleField* createCudaParticleField(int deviceId, int numValues, int numVectors, math::vec4 start, math::vec4 length, math::vec4 size)
{
	if (deviceId >= 0)
	{
		return new CudaParticleField(deviceId, numValues, numVectors, start, length, size);
	}
	else
	{
		return new CudaParticleField(numValues, numVectors, start, length, size);
	}
}

} /* namespace partflow */
} /* namespace PFCore */
