/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <vrbase/scenes/render/BasicRenderedScene.h>
#include <string>

namespace vrbase {

BasicRenderedScene::BasicRenderedScene(SceneRef scene) : SceneAdapter(scene) {
}

BasicRenderedScene::~BasicRenderedScene() {
}

void BasicRenderedScene::init() {
	getInnerScene()->init();
	// Read source code from shader files
	std::string vs_text =
			"#version 330\n"
			"layout(location = 0) in vec3 pos;\n"
			"layout(location = 1) in vec3 normal;\n"
			"layout(location = 2) in vec3 loc;\n"
			"uniform mat4 Model;\n"
			"uniform mat4 View;\n"
			"uniform mat4 Projection;\n"
			"\n"
			"out vec3 p;\n"
			"out vec3 v;\n"
			"out vec3 N;\n"
			"\n"
			"void main() {\n"
			"	//p = (Model * vec4(pos,1.0)).xyz;\n"
			"p = pos*25.0;\n"
			"v = (View * Model * vec4(pos+loc,1.0)).xyz;\n"
			"gl_Position = Projection*View*Model*vec4(pos+loc,1.0);\n"
			"N = normalize((View*Model*vec4(normal,0)).xyz);\n"
			"}\n";
			/*
    "#version 330\n"
    "layout(location = 0) in vec3 position;"
    "layout(location = 1) in vec3 normal;"
    "uniform mat4 model;"
    "uniform mat4 view;"
    "uniform mat4 proj;"
    "out vec3 fragColor, fragPosition, fragNormal;\n"
    "void main() {\n"
    "  fragColor = vec4(1.0,0.0,0.0,1.0);\n"
    "  gl_Position = proj * view * model * vec4(position,1.0);\n"
    "  fragPosition = (model * vec4(position,1.0)).xyz;\n"
    "  fragNormal = normalize((transpose(inverse(model)) * vec4(normal,1.0)).xyz);\n"
    "}\n";*/

    std::string fs_text =
    		"#version 330\n"
    		"layout(location = 0) out vec4 FragColor;\n"
    		"\n"
    		"uniform mat4 View;\n"
    		"in vec3 N;\n"
    		"in vec3 v;\n"
    		"in vec3 p;\n"
    		"\n"
    		"void main() {\n"
    		//"	FragColor = vec4(1.0,0,0,1.0);\n"

    		"	vec3 lightPos = vec3(0.5, 0.0, 3.0);\n"
    		"	vec4 color = vec4(1.0,0,0,1.0);\n" //vec4(p.y,0.4,1.0,0.3);\n"
    		"	vec3 L = normalize(lightPos - v); \n"
    		"\n"
    		"    //calculate Ambient Term:\n"
    		"    vec4 Iamb = vec4(0.2, 0.2, 0.2, 1.0);\n"
    		"    //calculate Diffuse Term:\n"
    		"    vec4 Idiff = max(dot(N,-L), 0.0) * vec4(0.7, 0.7, 0.7, 1.0);\n"
    		"    // write Total Color:\n"
    		"    FragColor = (Idiff + Iamb) * color;\n"
    		"}\n";

    _shader = ShaderRef(new Shader(vs_text, fs_text));
    _shader->init();
}

void BasicRenderedScene::draw(const Camera& camera) {
	_shader->useProgram();
	_shader->setParameter("Model", camera.getObjectToWorldMatrix());
	_shader->setParameter("View", camera.getViewMatrix());
	_shader->setParameter("Projection", camera.getProjetionMatrix());
	getInnerScene()->draw(camera);
}

} /* namespace vrbase */
