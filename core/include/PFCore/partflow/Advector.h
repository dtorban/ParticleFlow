/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef ADVECTOR_H_
#define ADVECTOR_H_

#include "PFCore/partflow/ParticleSetView.h"

namespace PFCore {
namespace partflow {

class Advector {
public:
	virtual ~Advector() {}

	virtual void advectParticles(ParticleSetView& particleSet, int step, float time, float dt, int iterations = 1) = 0;
};

} /* namespace partflow */
} /* namespace PFCore */

#endif /* ADVECTOR_H_ */
