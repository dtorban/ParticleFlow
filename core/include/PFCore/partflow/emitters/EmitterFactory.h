/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef EMITTERFACTORY_H_
#define EMITTERFACTORY_H_

#include "PFCore/partflow/emitters/BasicEmitter.h"
#include "PFCore/partflow/emitters/strategies/SphereEmitter.h"

namespace PFCore {
namespace partflow {

class EmitterFactory {
public:
	EmitterFactory() {}
	virtual ~EmitterFactory() {}

	virtual Emitter* createSphereEmitter(math::vec3 pos, float radius, int duration)
	{
		return new BasicEmitter<SphereEmitter>(SphereEmitter(pos, radius, duration));
	}
};

} /* namespace partflow */
} /* namespace PFCore */

#endif /* EMITTERFACTORY_H_ */
