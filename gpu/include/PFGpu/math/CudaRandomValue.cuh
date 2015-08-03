/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef CUDARANDOMVALUE_H_
#define CUDARANDOMVALUE_H_

#include "PFCore/env_cuda.h"
#include "PFCore/math/RandomValue.h"

namespace PFCore {
namespace math {

struct CudaRandomValue : public RandomArrayValue {
	CudaRandomValue(int deviceId, int size);
	virtual ~CudaRandomValue();
	
private:
	int _deviceId;
};

} /* namespace math */
} /* namespace PFCore */

#endif /* CUDARANDOMVALUE_H_ */
