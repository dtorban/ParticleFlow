/*
 * ParticleSceneUpdater.h
 *
 *  Created on: Aug 16, 2015
 *      Author: dtorban
 */

#ifndef SOURCE_DIRECTORY__VIS_PARTFLOW_INCLUDE_PFVIS_SCENES_UPDATE_PARTICLESCENEUPDATER_H_
#define SOURCE_DIRECTORY__VIS_PARTFLOW_INCLUDE_PFVIS_SCENES_UPDATE_PARTICLESCENEUPDATER_H_

#include "PFCore/partflow/PartflowRef.h"

namespace PFVis {
namespace partflow {

class ParticleSceneUpdater {
public:
	virtual ~ParticleSceneUpdater() {}

	virtual void updateParticleSet(PFCore::partflow::ParticleSetRef particleSet) = 0;
};

} /* namespace partflow */
} /* namespace PFVis */

#endif /* SOURCE_DIRECTORY__VIS_PARTFLOW_INCLUDE_PFVIS_SCENES_UPDATE_PARTICLESCENEUPDATER_H_ */
