/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef PARTFLOWREF_H_
#define PARTFLOWREF_H_

#include <memory>
#include "PFCore/partflow/Advector.h"
#include "PFCore/partflow/Emitter.h"
#include "PFCore/partflow/ParticleSet.h"
#include "PFCore/partflow/vectorFields/ParticleField.h"
#include "PFCore/partflow/ParticleUpdater.h"

namespace PFCore {
namespace partflow {

typedef std::shared_ptr<Advector> AdvectorRef;
typedef std::shared_ptr<Emitter> EmitterRef;
typedef std::shared_ptr<ParticleSet> ParticleSetRef;
typedef std::shared_ptr<ParticleField> ParticleFieldRef;
typedef std::shared_ptr<ParticleUpdater> ParticleUpdaterRef;

} /* namespace partflow */
} /* namespace PFCore */

#endif /* PARTFLOWREF_H_ */
