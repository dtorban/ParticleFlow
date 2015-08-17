/*
 * CudaResource.h
 *
 *  Created on: Aug 16, 2015
 *      Author: dtorban
 */

#ifndef SOURCE_DIRECTORY__GPU_INCLUDE_CUDARESOURCE_H_
#define SOURCE_DIRECTORY__GPU_INCLUDE_CUDARESOURCE_H_

#include "GpuResource.h"

namespace PFCore {

class CudaResource : public GpuResource {
public:
	CudaResource(int deviceId, int resourceId);
	virtual ~CudaResource();
	
	void map(void*& data);
	void unmap();
	
private:
	int _deviceId;
	int _resourceId;
	struct cudaGraphicsResource *resource;
	//cudaGraphicsResource *resource[1];
};

} /* namespace PFCore */

#endif /* SOURCE_DIRECTORY__GPU_INCLUDE_CUDARESOURCE_H_ */
