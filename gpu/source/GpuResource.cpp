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

bool GpuResource::map()
{
	return false;
}

int GpuResource::getData(void** data)
{
	return 0;
}

void GpuResource::unmap()
{
}

int GpuResource::getDeviceId()
{
	return -1;
}

} /* namespace PFCore */
