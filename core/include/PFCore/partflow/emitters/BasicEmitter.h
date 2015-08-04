/*
 * BasicEmitter.h
 *
 *  Created on: Jul 28, 2015
 *      Author: dtorban
 */

#ifndef SOURCE_DIRECTORY__CORE_INCLUDE_PFCORE_PARTFLOW_EMITTERS_BASICEMITTER_H_
#define SOURCE_DIRECTORY__CORE_INCLUDE_PFCORE_PARTFLOW_EMITTERS_BASICEMITTER_H_

#include "PFCore/partflow/Emitter.h"
#include "PFCore/math/RandomValue.h"
#include <iostream>

namespace PFCore {
namespace partflow {

template<typename Strategy>
class BasicEmitter : public Emitter {
public:
	BasicEmitter(const Strategy &strategy);
	virtual ~BasicEmitter();

	virtual void emitParticles(ParticleSetView& particleSet, int step, bool init = false);

protected:
	Strategy _strategy;
};

template<typename Strategy>
inline BasicEmitter<Strategy>::BasicEmitter(
		const Strategy &strategy) : _strategy(strategy) {
}

template<typename Strategy>
inline BasicEmitter<Strategy>::~BasicEmitter() {
}

template<typename Strategy>
inline void BasicEmitter<Strategy>::emitParticles(
		ParticleSetView& particleSet, int step, bool init) {

	for (int index = 0; index < particleSet.getNumParticles(); index++)
	{
		_strategy.emitParticle(particleSet, index, step, math::RandomValue(), init);
	}
}

} /* namespace partflow */
} /* namespace PFCore */



#endif /* SOURCE_DIRECTORY__CORE_INCLUDE_PFCORE_PARTFLOW_EMITTERS_BASICEMITTER_H_ */
