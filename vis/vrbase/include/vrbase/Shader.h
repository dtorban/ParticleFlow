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
#include <memory>
#include "vrbase/GraphicsObject.h"

namespace vrbase {

class Shader;
typedef std::shared_ptr<class Shader> ShaderRef;

class Shader : public ContextObject {
public:
	Shader(const std::string &vertexShader, const std::string &geometryShader, const std::string &fragmentShader);
	Shader(const std::string &vertexShader, const std::string &fragmentShader);
	virtual ~Shader();

	void init();
	void useProgram();
	void releaseProgram();
	void setParameter(const std::string& name, glm::mat4 matrix);
	void setParameter(const std::string& name, glm::vec3 vector);
	void setParameter(const std::string& name, GLuint id);
	void setParameter(const std::string& name, float* values, int numValues);

	void initContextItem();
	void destroyContextItem();

private:
	std::string loadFile(const std::string &fileName);
	void compileShader(const GLuint &shader, const std::string& code);
	bool checkShaderCompileStatus(GLuint obj);
	bool checkProgramLinkStatus(GLuint obj);

	bool _isInitialized;
	ContextSpecificPtr<GLuint> _shaderProgram;
	std::string _vertexShader;
	std::string _geometryShader;
	std::string _fragmentShader;
};

}


#endif /* SHADER_H_ */
