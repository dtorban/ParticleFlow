/*
 * GpuResourceFactory.h
 *
 *  Created on: Aug 16, 2015
 *      Author: dtorban
 */

#ifndef SOURCE_DIRECTORY__GPU_INCLUDE_PFGPU_GPURESOURCEFACTORY_H_
#define SOURCE_DIRECTORY__GPU_INCLUDE_PFGPU_GPURESOURCEFACTORY_H_

#include "GpuResource.h"

namespace PFCore {

class GpuResourceFactory {
public:
	GpuResourceFactory(int deviceId);
	virtual ~GpuResourceFactory();

	virtual GpuResource* registerResource(int resourceId);

private:
	int _deviceId;
};

} /* namespace PFCore */

#endif /* SOURCE_DIRECTORY__GPU_INCLUDE_PFGPU_GPURESOURCEFACTORY_H_ */
