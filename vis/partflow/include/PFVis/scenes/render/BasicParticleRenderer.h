/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef BASICPARTICLERENDERER_H_
#define BASICPARTICLERENDERER_H_

#include "vrbase/scenes/render/BasicRenderedScene.h"

namespace PFCore {
namespace partflow {

class BasicParticleRenderer : public vrbase::BasicRenderedScene {
public:
	BasicParticleRenderer(vrbase::SceneRef scene);
	BasicParticleRenderer(vrbase::SceneRef scene, vrbase::ShaderRef shader);
	virtual ~BasicParticleRenderer();

private:
	vrbase::ShaderRef createBasicShader();
};

} /* namespace partflow */
} /* namespace PFCore */

#endif /* BASICPARTICLERENDERER_H_ */
