/*
 * GpuResource.cpp
 *
 *  Created on: Aug 16, 2015
 *      Author: dtorban
 */

#include <PFGpu/GpuResource.h>

namespace PFCore {

#ifndef USE_CUDA
extern "C"
GpuResource* gpuRegisterResource(int deviceId, int resourceId)
{
	return new GpuResource();
}
#endif

GpuResource::GpuResource() {
}

GpuResource::~GpuResource() {
}

void GpuResource::map(void*& data)
{
}

void GpuResource::unmap()
{
}

} /* namespace PFCore */
