/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <PFVis/scenes/render/BasicParticleRenderer.h>
#include <sstream>

using namespace PFCore::partflow;

namespace PFVis {
namespace partflow {

BasicParticleRenderer::BasicParticleRenderer(vrbase::SceneRef scene, const PFCore::partflow::ParticleSetView& particleSet) : vrbase::BasicRenderedScene(scene, createBasicShader(particleSet)) {
}

BasicParticleRenderer::BasicParticleRenderer(vrbase::SceneRef scene, vrbase::ShaderRef shader) : vrbase::BasicRenderedScene(scene, shader) {
}

BasicParticleRenderer::~BasicParticleRenderer() {
	// TODO Auto-generated destructor stub
}

vrbase::ShaderRef BasicParticleRenderer::createBasicShader(const PFCore::partflow::ParticleSetView& particleSet) {
	int numSteps = particleSet.getNumSteps();

	int loc = 0;

	std::stringstream ss;
	ss << "#version 330\n";
	ss << "layout(location = 0) in vec3 pos;\n";
	ss << "layout(location = 1) in vec3 normal;\n";
	loc = 2;
	ss << "layout(location = " << loc << ") in vec3 loc[];\n";
	loc += numSteps;
	if (particleSet.getNumAttributes() > 0)
	{
		ss << "layout(location = " << loc << ") in int attributes[];\n";
		loc += numSteps*particleSet.getNumAttributes();
	}
	if (particleSet.getNumValues() > 0)
	{
		ss << "layout(location = " << loc << ") in float values[];\n";
		loc += numSteps*particleSet.getNumValues();
	}
	if (particleSet.getNumVectors() > 0)
	{
		ss << "layout(location = " << loc << ") in vec3 vectors[];\n";
	}
	ss << "uniform mat4 Model;\n" <<
			"uniform mat4 View;\n" <<
			"uniform mat4 Projection;\n" <<
			"\n" <<
			"out vec3 p;\n" <<
			"out vec3 v;\n" <<
			"out vec3 N;\n" <<
			"out float mag;\n" <<
			"out vec3 velocity;\n" <<
			"\n" <<
			"void main() {\n" <<
			"mag = length(vectors[0])/90.0;\n" <<
			"velocity = vectors[0];\n" <<
			"v = (View * Model * vec4((pos)*mag+loc[0],1.0)).xyz;\n";
	if (numSteps > 1)
	{
		ss << "if (pos.x < 0.0) {v = (View * Model * vec4((pos)*mag+loc[0],1.0)).xyz;}\n" <<
				"else if (pos.x > 0.0) {v = (View * Model * vec4((pos)*mag+loc[1],1.0)).xyz;}\n";
	}
	//ss << "gl_Position = Projection*View*Model*vec4((pos)*mag+loc[0],1.0);\n";
	if (numSteps > 1)
	{
		ss << "if (pos.x < 0.0) {gl_Position = Projection*View*Model*vec4((pos)*mag+loc[0],1.0);}\n" <<
				"else if (pos.x > 0.0) {gl_Position = Projection*View*Model*vec4((pos)*mag+loc[1],1.0);}\n" <<
				"else {\n";
	}
	ss << "gl_Position = Projection*View*Model*vec4((pos)*mag+loc[0],1.0);\n";
	if (numSteps > 1)
	{
		ss << "}\n";
	}
	ss << "N = normalize((View*Model*vec4(normal,0)).xyz);\n" <<
			"}\n";

	std::string vs_text = ss.str();
			/*"#version 330\n"
			"layout(location = 0) in vec3 pos;\n"
			"layout(location = 1) in vec3 normal;\n"
			"layout(location = 2) in vec3 loc;\n"
			//"layout(location = 3) in float attributes[];\n"
			//"layout(location = 3) in float values[];\n"
			"layout(location = 3) in vec3 vectors[];\n"
			"uniform mat4 Model;\n"
			"uniform mat4 View;\n"
			"uniform mat4 Projection;\n"
			"\n"
			"out vec3 p;\n"
			"out vec3 v;\n"
			"out vec3 N;\n"
			"out float mag;\n"
			"out vec3 velocity;\n"
			"\n"
			"void main() {\n"
			"mag = length(vectors[0])/90.0;\n"
			"velocity = vectors[0];\n"
			"v = (View * Model * vec4((pos)*mag+loc,1.0)).xyz;\n"
			"gl_Position = Projection*View*Model*vec4((pos)*mag+loc,1.0);\n"
			"N = normalize((View*Model*vec4(normal,0)).xyz);\n"
			"}\n";*/

    std::string fs_text =
    		"#version 330\n"
    		"layout(location = 0) out vec4 FragColor;\n"
    		"\n"
    		"uniform mat4 View;\n"
    		"in vec3 N;\n"
    		"in vec3 v;\n"
    		"in vec3 p;\n"
    		"in float mag;\n"
			"in vec3 velocity;\n"
    		"\n"
    		"void main() {\n"
    		"	if (mag < 0.001) discard;\n"
    		"	vec3 lightPos = vec3(0.5, 0.0, 3.0);\n"
    		"	vec4 color = vec4(mag, 1.0-mag, 0.0, 1.0);\n"
    		"	vec3 L = normalize(lightPos - v); \n"
    		"\n"
    		"    //calculate Ambient Term:\n"
    		"    vec4 Iamb = vec4(0.2, 0.2, 0.2, 1.0);\n"
    		"    //calculate Diffuse Term:\n"
    		"    vec4 Idiff = max(dot(N,-L), 0.0) * vec4(0.7, 0.7, 0.7, 1.0);\n"
    		"    // write Total Color:\n"
    		"    FragColor = (Idiff + Iamb) * color;\n"
    		"}\n";

    return vrbase::ShaderRef(new vrbase::Shader(vs_text, fs_text));
}

} /* namespace partflow */
} /* namespace PFVis */
