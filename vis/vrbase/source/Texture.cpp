/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/lgpl-3.0.html.
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include "vrbase/Texture.h"

namespace vrbase {

Texture::Texture() : _id(0), _loaded(false), _tex(NULL), _width(0), _height(0), _depth(0) {
}

Texture::Texture(int width, int height, int depth) : _id(0), _loaded(false), _tex(NULL) {
	create(width, height, depth);
}

Texture::~Texture() {
	if (_tex != NULL)
	{
		delete [] _tex;
	}
	if (_loaded)
	{
		glDeleteTextures(1, &_id);
	}
}

void Texture::create(int width, int height, int depth) {

	_width = width;
	_height = height;
	_depth = depth;

	if (_tex != NULL)
	{
		delete [] _tex;
	}

	_tex = new GLubyte[width*height*depth*4];
	_loaded = false;
}

void Texture::load() {

	if (_tex == NULL)
	{
		create(0,0,0);
	}

	glGenTextures(1, &_id);

	if (_depth > 1)
	{
		 glEnable(GL_TEXTURE_3D);
		 //glGenTextures(1, &_id);
	    /*glBindTexture(GL_TEXTURE_3D, _id);
	    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
	    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, _width, _height, _depth, 0, GL_RGBA, GL_UNSIGNED_BYTE, _tex);*/

		glBindTexture(GL_TEXTURE_3D, _id);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
		glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, _width, _height, _depth, 0, GL_RGBA, GL_UNSIGNED_BYTE, _tex);
	}
	else
	{
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, _id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _width, _height, 0, GL_RGBA, GL_UNSIGNED_BYTE, _tex);
	}

	delete [] _tex;

	_loaded = true;
}

}

