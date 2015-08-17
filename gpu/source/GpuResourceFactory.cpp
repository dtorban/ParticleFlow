/*
 * GpuResourceFactory.cpp
 *
 *  Created on: Aug 16, 2015
 *      Author: dtorban
 */

#include <PFGpu/GpuResourceFactory.h>

namespace PFCore {

extern "C"
GpuResource* gpuRegisterResource(int deviceId, int resourceId);

GpuResourceFactory::GpuResourceFactory(int deviceId) : _deviceId(deviceId) {
}

GpuResourceFactory::~GpuResourceFactory() {
}

GpuResource* GpuResourceFactory::registerResource(int resourceId) {
	if (_deviceId < 0)
	{
		return new GpuResource();
	}
	else
	{
		return gpuRegisterResource(_deviceId, resourceId);
	}
}

} /* namespace PFCore */
