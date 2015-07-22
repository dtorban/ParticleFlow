/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <PFCore/input/loaders/BrickOfFloatLoader.h>
#include <stdio.h>

namespace PFCore {
namespace input {

BrickOfFloatLoader::BrickOfFloatLoader(const std::string &fileName) :
		_fileName(fileName), _xsize(0), _ysize(0), _zsize(0), _tsize(0), _skip(
				0) {
	//_values[x + z*_size.x + y*_size.x*_size.z + t*_size.x*_size.y*_size.z)];
}

BrickOfFloatLoader::BrickOfFloatLoader(const std::string &fileName, int xsize,
		int ysize, int zsize, int tsize, int skip) :
		_fileName(fileName), _xsize(xsize), _ysize(ysize), _zsize(zsize), _tsize(
				tsize), _skip(skip) {

}

int BrickOfFloatLoader::getLocation(int x, int y, int z, int t) {
	return (x + y * _xsize + z * _xsize * _ysize + t * _xsize * _ysize * _zsize)
			* sizeof(float);
}

BrickOfFloatLoader::~BrickOfFloatLoader() {
}

bool BrickOfFloatLoader::load(float* data, int size) {

	size_t len_read = 0;

	FILE *fp;
	printf("%s\n", _fileName.c_str());
	fp = fopen(_fileName.c_str(), "rb");

	if (!fp) {
		return false;
	}

	if (_skip > 0) {
		float* data_pos = data;

		for (int z = 0; z < 100; z += _skip) {
			for (int y = 0; y < 500; y += _skip) {
				for (int x = 0; x < 500; x += _skip) {
					fseek(fp, getLocation(x, y, z, 0), SEEK_SET);
					len_read = fread((float *) data_pos, (1 * sizeof(float)), 1,
							fp);
					data_pos++;
				}
			}
		}
	} else {
		len_read = fread((float *) data, (size * sizeof(float)), 1, fp);
		if (!len_read) {
			fclose(fp);
			return false;
		}
	}

	int i, j, k;
	k = 0;

	for (int f = 0; f < size; f++) {
		float newFloat;
		char *floatToConvert = (char*) &data[f];
		char *returnFloat = (char*) &newFloat;
		returnFloat[0] = floatToConvert[3];
		returnFloat[1] = floatToConvert[2];
		returnFloat[2] = floatToConvert[1];
		returnFloat[3] = floatToConvert[0];
		data[f] = newFloat;

		if (data[f] < 1.0e35) {
			//printf("%f\n", data[f]);
		}
	}

	fclose(fp);

	return true;
}

}}
