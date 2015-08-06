/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/lgpl-3.0.html.
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef SHADER_H_
#define SHADER_H_

#include <string>
#include "GL/glew.h"
#include <glm/glm.hpp>
#include <memory>

namespace vrbase {

class Shader;
typedef std::shared_ptr<class Shader> ShaderRef;

class Shader {
public:
	Shader(const std::string &vertexShader, const std::string &geometryShader, const std::string &fragmentShader);
	Shader(const std::string &vertexShader, const std::string &fragmentShader);
	virtual ~Shader();

	void init();
	void useProgram();
	void setParameter(const std::string& name, glm::mat4 matrix);
	void setParameter(const std::string& name, glm::vec3 vector);
	void setParameter(const std::string& name, GLuint id);

private:
	std::string loadFile(const std::string &fileName);
	void compileShader(const GLuint &shader, const std::string& code);
	bool checkShaderCompileStatus(GLuint obj);
	bool checkProgramLinkStatus(GLuint obj);

	bool _isInitialized;
	GLuint _shaderProgram;
	std::string _vertexShader;
	std::string _geometryShader;
	std::string _fragmentShader;
};

}


#endif /* SHADER_H_ */
