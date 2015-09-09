/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef CUDAVECTORFIELDADVECTOR_H_
#define CUDAVECTORFIELDADVECTOR_H_

#include "PFCore/env_cuda.h"
#include "PFCore/partflow/Advector.h"
#include <string>
#include <iostream>
#include <PFCore/partflow/advectors/strategies/EulerAdvector.h>
#include <PFCore/partflow/advectors/strategies/RungaKutta4.h>
#include <PFCore/partflow/vectorFields/ConstantField.h>
#include <PFCore/partflow/vectorFields/ParticleFieldVolume.h>

namespace PFCore {
namespace partflow {

template<typename Strategy, typename VField>
class CudaVectorFieldAdvector : public Advector {
public:
	CudaVectorFieldAdvector(void* strategy, void* vectorField);
	virtual ~CudaVectorFieldAdvector();
	
	void advectParticles(ParticleSetView& particleSet, int step, float time, float dt, int iterations);
	
private:
	Strategy _strategy;
	VField _vectorField;
};

template<typename Strategy, typename VField>
inline CudaVectorFieldAdvector<Strategy, VField>::CudaVectorFieldAdvector(void* strategy, void* vectorField) {
	_strategy = *(reinterpret_cast<Strategy*>(strategy));
	_vectorField = *(reinterpret_cast<VField*>(vectorField));
}

template<typename Strategy, typename VField>
inline CudaVectorFieldAdvector<Strategy, VField>::~CudaVectorFieldAdvector() {
}

template<typename Strategy, typename VField>
__global__ void CudaVectorFieldAdvector_advectParticle(Strategy strategy, VField vectorField, ParticleSetView particleSet, int step, int prevStep, float time, float dt, int iterations)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < particleSet.getNumParticles())
	{
		strategy.advectParticle(particleSet, vectorField, i, step, prevStep, time, dt, iterations);
	}
}


template<typename Strategy, typename VField>
void CudaVectorFieldAdvector<Strategy, VField>::advectParticles(ParticleSetView& particleSet, int step, float time, float dt, int iterations) {
	//std::cout << "Advect cuda!" << std::endl;
	int prevStep = (particleSet.getNumSteps() + step - 1) % particleSet.getNumSteps();
	CudaVectorFieldAdvector_advectParticle<Strategy, VField><<<1024, 512>>>(_strategy, _vectorField, particleSet, step, prevStep, time, dt, iterations);
	cudaDeviceSynchronize();
}

template<typename VField>
inline Advector* createCudaVectorFieldAdvectorFromField(std::string strategyTypeId, void* strategy, void* vectorField)
{
	if (strategyTypeId == "Euler")
	{
		return new CudaVectorFieldAdvector<EulerAdvector<VField>, VField>(strategy, vectorField);
	}
	else if (strategyTypeId == "RungaKutta4")
	{
		return new CudaVectorFieldAdvector<RungaKutta4<VField>, VField>(strategy, vectorField);
	}
	
	return NULL;
}

inline Advector* parseCudaVectorFieldAdvector(std::string strategyTypeId, std::string vectorFieldTypeId, void* strategy, void* vectorField)
{
	if (vectorFieldTypeId == "ConstantField")
	{
		return createCudaVectorFieldAdvectorFromField<ConstantField>(strategyTypeId, strategy, vectorField);
	}
	else if (vectorFieldTypeId == "ParticleFieldVolume")
	{
		return createCudaVectorFieldAdvectorFromField<ParticleFieldVolume>(strategyTypeId, strategy, vectorField);
	}
	
	return NULL;
}

} /* namespace partflow */
} /* namespace PFCore */

#endif /* CUDAVECTORFIELDADVECTOR_H_ */
