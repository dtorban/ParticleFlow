/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef EMITTER_H_
#define EMITTER_H_

#include "PFCore/partflow/ParticleSetView.h"

namespace PFCore {
namespace partflow {

class Emitter {
public:
	virtual ~Emitter() {}

	virtual void emitParticles(ParticleSetView& particleSet, int step, bool init = false) = 0;
};

} /* namespace partflow */
} /* namespace PFCore */

#endif /* EMITTER_H_ */
