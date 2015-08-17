/*
 * GpuResource.h
 *
 *  Created on: Aug 16, 2015
 *      Author: dtorban
 */

#ifndef SOURCE_DIRECTORY__GPU_INCLUDE_PFGPU_GPURESOURCE_H_
#define SOURCE_DIRECTORY__GPU_INCLUDE_PFGPU_GPURESOURCE_H_

namespace PFCore {

class GpuResource {
public:
	GpuResource();
	virtual ~GpuResource();

	virtual void map(void*& data);
	virtual void unmap();
};

} /* namespace PFCore */

#endif /* SOURCE_DIRECTORY__GPU_INCLUDE_PFGPU_GPURESOURCE_H_ */
