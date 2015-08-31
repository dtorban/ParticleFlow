/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <PFVis/scenes/render/BasicParticleRenderer.h>
#include <sstream>
#include <math.h>

using namespace PFCore::partflow;

namespace PFVis {
namespace partflow {

BasicParticleRenderer::BasicParticleRenderer(vrbase::SceneRef scene, const PFCore::partflow::ParticleSetView& particleSet, int *currentStep, float* shape) : vrbase::BasicRenderedScene(scene, createBasicShader(particleSet)), _currentStep(currentStep), _shape(shape) {
}

BasicParticleRenderer::BasicParticleRenderer(vrbase::SceneRef scene, vrbase::ShaderRef shader, int *currentStep, float* shape) : vrbase::BasicRenderedScene(scene, shader), _currentStep(currentStep), _shape(shape) {
}

BasicParticleRenderer::~BasicParticleRenderer() {
}

vrbase::ShaderRef BasicParticleRenderer::createBasicShader(const PFCore::partflow::ParticleSetView& particleSet) {
	_numSteps = particleSet.getNumSteps();

	int numSteps = 2;//particleSet.getNumSteps()-1;

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
	ss << "uniform float shape[" << (_numSteps > 1 ? _numSteps : _numSteps+1) << "];\n";
	ss << "uniform mat4 Model;\n" <<
			"uniform mat4 View;\n" <<
			"uniform mat4 Projection;\n" <<
			"uniform int currentStep;\n" <<
			"uniform int positionOffset;\n" <<
			"\n" <<
			"out vec3 p;\n" <<
			"out vec3 v;\n" <<
			"out vec3 N;\n" <<
			"out float mag;\n" <<
			"flat out int numSteps;\n" <<
			"out vec3 velocity;\n" <<
			"flat out int InstanceID;\n" <<
			"flat out int numParticles;\n" <<
			"flat out int valid;\n" <<
			"\n" <<
			"void main() {\n" <<
			"numParticles = " << particleSet.getNumParticles() << ";\n" <<
			"numSteps = " << particleSet.getNumSteps()  << ";\n" <<
			"InstanceID = gl_InstanceID;\n" <<
			"vec3 vertLoc = loc[0];\n" <<
			"vec3 vert = vec3(pos.xy, 0);\n" <<
			"if (length(loc[0]-loc[1]) > 50) {valid = 0;} else {valid = 1;}\n" <<

			"int step = InstanceID/numParticles;\n" <<

			"if (pos.z > 0.0f) {\n" <<
			"	mag = length(vectors[1])/90.0;\n" <<
			"	velocity = vectors[1];\n" <<
			"	vertLoc = loc[1];\n" <<
			"   step = step - 1;\n" <<
			"   if (numSteps == 1)\n" <<
			"	{\n" <<
			"		vertLoc = loc[0]-vectors[0]/2.0;\n" <<
			"		mag = length(vectors[0])/90.0;\n" <<
			"		velocity = vectors[0];\n" <<
			//"   	step = step + 1;\n" <<
			"	}\n" <<
			"} else {\n" <<
			"	mag = length(vectors[0])/90.0;\n" <<
			"	velocity = vectors[0];\n" <<
			"}\n" <<


			"float size = 1.0;\n" <<
			"int steps = 0;\n" <<
			"if (step < mod(currentStep + 1 - positionOffset, numSteps)) { steps = currentStep + 1 - positionOffset - step; }\n" <<
			"else { steps = numSteps + currentStep + 1 - positionOffset - step; }\n" <<
			"if (steps-1 == 0) { size = 0.2;\n }" <<
			"else if (steps-1 == 1) { size = 0.8;\n }" <<
			"else if (steps-1 == 2) { size = 0.9;\n }" <<
			"else { size = 1.0 - 1.0*steps/numSteps;\n }" <<
			//"size = 1.0;\n" <<
			"size = shape[steps - 1];\n" <<
			"if (numSteps == 1) size = shape[0];\n" <<
			"if (numSteps == 1 && step < 0) size = shape[1];\n" <<

			"vec3 zp = normalize(velocity);\n" <<
			"vec3 y = vec3(0,1,0);\n" <<
			"if (zp.x > 0) y = y*-1;\n" <<
			"vec3 yp = normalize(y-zp*(dot(y,zp)));\n" <<
			"vec3 xp = cross(yp,zp);\n" <<
			"mat4 rot = mat4(vec4(xp,0),vec4(yp,0),vec4(zp,0),vec4(0,0,0,1));\n" <<
			"v = (View * Model *vec4((rot*vec4(vert,1.0)).xyz*mag*size+vertLoc,1.0)).xyz;\n";
	ss << "gl_Position = Projection*View*Model* vec4((rot*vec4(vert,1.0)).xyz*mag*size+vertLoc,1.0);\n";
	ss << "N = normalize((View* Model*rot*vec4(normal,0)).xyz);\n" <<
			"}\n";
	std::string vs_text = ss.str();

    std::string fs_text =
    		"#version 330\n"
    		"layout(location = 0) out vec4 FragColor;\n"
    		"\n"
    		"uniform mat4 View;\n"
			"uniform int currentStep;\n"
			"uniform int positionOffset;\n"
    		"in vec3 N;\n"
    		"in vec3 v;\n"
    		"in vec3 p;\n"
    		"in float mag;\n"
    		"flat in int numSteps;\n"
			"in vec3 velocity;\n"
    		"flat in int InstanceID;\n"
			"flat in int numParticles;\n"
			"flat in int valid;\n"
    		"\n"
    		"void main() {\n"
    		"	if (mag < 0.001 || valid == 0) discard;\n"
    		"   int step = InstanceID/numParticles;\n"
    		"	if (numSteps > 1 && mod(currentStep + 1 - positionOffset, numSteps) == step) discard;\n"
    		"	vec3 lightPos = vec3(0.5, 0.0, 3.0);\n"
    		"	vec4 color = vec4(mag, 1.0-mag, 0.0, 1.0);\n"
    		"	vec3 L = normalize(lightPos - v); \n"
    		"\n"
    		"    //calculate Ambient Term:\n"
    		"    vec4 Iamb = vec4(0.2, 0.2, 0.2, 1.0);\n"
    		"    //calculate Diffuse Term:\n"
    		"    vec4 Idiff = max(dot(N,-L), 0.0) * vec4(0.7, 0.7, 0.7, 1.0);\n"
    		//"    vec4 Ispec = max(dot(N,-L), 0.0) * vec4(0.7, 0.7, 0.7, 1.0);\n"
    		"    // write Total Color:\n"
    		"    FragColor = (Idiff + Iamb) * color;\n"
    		"}\n";

    return vrbase::ShaderRef(new vrbase::Shader(vs_text, fs_text));
}

void BasicParticleRenderer::setShaderParameters(const vrbase::Camera& camera,
		vrbase::ShaderRef shader) {
	vrbase::BasicRenderedScene::setShaderParameters(camera, shader);

	shader->setParameter("currentStep", (*_currentStep)%_numSteps);

	shader->setParameter("shape", _shape, _numSteps > 1 ? _numSteps : _numSteps+1);
}

} /* namespace partflow */
} /* namespace PFVis */
