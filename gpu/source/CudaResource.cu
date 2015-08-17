/*
 * CudaResource.cpp
 *
 *  Created on: Aug 16, 2015
 *      Author: dtorban
 */

#include <PFGpu/CudaResource.cuh>
#include <cuda_gl_interop.h>

namespace PFCore {

extern "C"
GpuResource* gpuRegisterResource(int deviceId, int resourceId)
{
	return new CudaResource(deviceId, resourceId);
}

CudaResource::CudaResource(int deviceId, int resourceId) : _deviceId(deviceId), _resourceId(resourceId) {
	cudaSetDevice(_deviceId);
	cudaGLSetGLDevice(_deviceId);
	cudaGraphicsGLRegisterBuffer(&resource, resourceId, cudaGraphicsMapFlagsNone);
}

CudaResource::~CudaResource() {
	cudaSetDevice(_deviceId);
	cudaGLSetGLDevice(_deviceId);
	cudaGraphicsUnregisterResource(resource);
}

void CudaResource::map(void*& data)
{
	cudaSetDevice(_deviceId);
	cudaGLSetGLDevice(_deviceId);
	cudaGraphicsMapResources(1, &resource);
	size_t size;
	cudaGraphicsResourceGetMappedPointer((void **)(&data), &size, resource);
}

void CudaResource::unmap()
{
	cudaSetDevice(_deviceId);
	cudaGLSetGLDevice(_deviceId);
	cudaGraphicsUnmapResources(1, &resource);
}

} /* namespace PFCore */
