/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <PFGpu/partflow/advectors/CudaVectorFieldAdvector.cuh>
#include <

namespace PFCore {
namespace partflow {

extern "C"
Advector* createCudaVectorFieldAdvector(std::string strategyTypeId, std::string vectorFieldTypeId, void* strategy, void* vectorField)
{
	if (strategyTypeId == "Euler" && vectorFieldTypeId == "ConstantField")
	{
		return new CudaVectorFieldAdvector<EulerAdvector<ConstantField>, ConstantField>(strategy, vectorField);
	}
	else if (strategyTypeId == "RungaKutta4" && vectorFieldTypeId == "ConstantField")
	{
		return new CudaVectorFieldAdvector<RungaKutta4<ConstantField>, ConstantField>(strategy, vectorField);
	}
}

} /* namespace partflow */
} /* namespace PFCore */
