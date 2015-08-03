/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <PFGpu/math/CudaRandomValue.cuh>

namespace PFCore {
namespace math {

CudaRandomValue::CudaRandomValue(int deviceId, int size) : RandomArrayValue(), _deviceId(deviceId) {
	numRand = size;
	cudaSetDevice(_deviceId);
	cudaMalloc(&rnd, numRand*sizeof(float));
}

CudaRandomValue::~CudaRandomValue() {
	cudaSetDevice(_deviceId);
	cudaFree(rnd);
}

/*extern "C"
RandomValue* createCudaRandomValue(int deviceId, int size)
{
	return new CudaRandomValue(deviceId, size);
}*/

} /* namespace math */
} /* namespace PFCore */
