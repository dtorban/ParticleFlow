/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef GPUVECTORFIELDADVECTOR_H_
#define GPUVECTORFIELDADVECTOR_H_

#include "PFCore/partflow/advectors/VectorFieldAdvector.h"
#include <string>

namespace PFCore {
namespace partflow {

#ifdef USE_CUDA
extern "C"
Advector* createCudaVectorFieldAdvector(std::string strategyTypeId, std::string vectorFieldTypeId, void* strategy, void* vectorField);
#endif

template<typename Strategy, typename VField>
class GpuVectorFieldAdvector : public Advector {
public:
	GpuVectorFieldAdvector(Strategy strategy, VField field);
	virtual ~GpuVectorFieldAdvector();

	void advectParticles(ParticleSetView& particleSet, int step, float time, float dt);

private:
	VectorFieldAdvector<Strategy, VField> _localAdvector;
#ifdef USE_CUDA
	Advector* _innerAdvector;
#endif
};

template<typename Strategy, typename VField>
inline GpuVectorFieldAdvector<Strategy, VField>::GpuVectorFieldAdvector(Strategy strategy, VField field) : _localAdvector(strategy, field)
{
#ifdef USE_CUDA
	_innerAdvector = createCudaVectorFieldAdvector(strategy.getTypeId(), field.getTypeId(), &strategy, &field);
#endif
}

template<typename Strategy, typename VField>
inline GpuVectorFieldAdvector<Strategy, VField>::~GpuVectorFieldAdvector() {
#ifdef USE_CUDA
	delete _innerAdvector;
#endif
}

template<typename Strategy, typename VField>
inline void GpuVectorFieldAdvector<Strategy, VField>::advectParticles(ParticleSetView& particleSet, int step, float time, float dt)
{
#ifdef USE_CUDA
	if (particleSet.getDeviceId() >= 0)
	{
		_innerAdvector->advectParticles(particleSet, step, time, dt);
		return;
	}
#endif

	_localAdvector.advectParticles(particleSet, step, time, dt);
}

}
}


#endif /* GPUVECTORFIELDADVECTOR_H_ */
