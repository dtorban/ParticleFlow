/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/lgpl-3.0.html.
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef TEXTURE_H_
#define TEXTURE_H_

#include <memory>
#include "GL/glew.h"

namespace vrbase {

class Texture;
typedef std::shared_ptr<class Texture> TextureRef;

class Texture {
public:
	Texture();
	Texture(int width, int height, int depth);
	virtual ~Texture();

	GLuint getId()
	{
		if (!_loaded)
		{
			load();
		}

		return _id;
	}

	inline GLubyte& get(int x, int y, int z, int index)
	{
		return _tex[((_height*x + y)*_depth + z)*4 + index];
	}

	inline GLubyte& get(int x, int y, int index)
	{
		return get(x, y, 0, index);
	}

	int getDepth() const {
		return _depth;
	}

	int getHeight() const {
		return _height;
	}

	int getWidth() const {
		return _width;
	}

protected:
	void create(int width, int height, int depth);
	virtual void load();

private:
	GLuint _id;
	GLubyte *_tex;
	int _width, _height, _depth;
	bool _loaded;
};

}

#endif /* TEXTURE_H_ */
