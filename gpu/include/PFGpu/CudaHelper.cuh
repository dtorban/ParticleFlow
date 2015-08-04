/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef CUDAHELPER_H_
#define CUDAHELPER_H_

#include "string.h"

namespace PFCore {

class CudaHelper {
public:
	static void copy(void* dst, int dstDeviceId, const void* src, int srcDeviceId, size_t size);
};

inline void CudaHelper::copy(void* dst, int dstDeviceId, const void* src, int srcDeviceId, size_t size)
{
	// Both local
	if (dstDeviceId < 0 && srcDeviceId < 0)
	{
		memcpy(dst, src, size);
	}
	// On same device
	else if (dstDeviceId == srcDeviceId)
	{
		cudaSetDevice(dstDeviceId);
		// TODO: kernal to copy to same device
		cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
	}
	// One of the devices is host
	else if (dstDeviceId < 0 || srcDeviceId < 0)
	{
		cudaSetDevice(dstDeviceId < 0 ? srcDeviceId : dstDeviceId);
		cudaMemcpy(dst, src, size, dstDeviceId < 0 ? cudaMemcpyDeviceToHost :  cudaMemcpyHostToDevice);
	}
	// Peer to peer
	else
	{
		cudaSetDevice(dstDeviceId);
		cudaMemcpyPeer(dst, dstDeviceId, src, srcDeviceId, size);
	}
}

} /* namespace PFCore */

#endif /* CUDAHELPER_H_ */
