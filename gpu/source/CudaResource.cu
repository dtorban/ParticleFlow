/*
 * CudaResource.cpp
 *
 *  Created on: Aug 16, 2015
 *      Author: dtorban
 */

#if defined(WIN32)
#define NOMINMAX
#include <windows.h>
#endif

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

bool CudaResource::map()
{
	cudaSetDevice(_deviceId);
	cudaGLSetGLDevice(_deviceId);
	cudaGraphicsMapResources(1, &resource);
	return true;
}

int CudaResource::getData(void** data)
{
	cudaSetDevice(_deviceId);
	cudaGLSetGLDevice(_deviceId);
	size_t size;
	cudaGraphicsResourceGetMappedPointer(data, &size, resource);
	return size;
}

void CudaResource::unmap()
{
	cudaSetDevice(_deviceId);
	cudaGLSetGLDevice(_deviceId);
	cudaGraphicsUnmapResources(1, &resource);
}

int CudaResource::getDeviceId()
{
	return _deviceId;
}

} /* namespace PFCore */
