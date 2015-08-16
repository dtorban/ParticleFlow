/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/lgpl-3.0.html.
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <vrbase/textures/DataTexture.h>
#include <glm/glm.hpp>
#include <iostream>

namespace vrbase {

DataTexture::DataTexture(int width, int height, int depth, float *data) : Texture() {
	// TODO Auto-generated constructor stub

	int res = 128;
	create(res, res, res);

	float max = 0.0f;

	for (int f = 0; f < width * height * depth; f++)
	{
		if (data[f] < 1.0e35)
		{
			max = glm::max(max, data[f]);
		}
	}

	std::cout << max << std::endl;
/*
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int z = 0; z < depth; z++)
			{

				float val = data[x + z*width + y*width*depth];
				if (val >= 1.0e35)
				{
					val = 0.0;
				}
				get(x,y,z,0) = 255;
				get(x,y,z,1) = 0;
				get(x,y,z,2) = 0;
				get(x,y,z,3) = 255;
			}
		}
	}
*/
	for (int i = 0; i < res; i++)
	{
		for (int j = 0; j < res; j++)
		{
			for (int k = 0; k < res; k++)
			{
				int a = (((float)(i))/((float)res))*width;
				int b = (((float)(j))/((float)res))*height;
				int c = (((float)(k))/((float)res))*depth;

				float val = data[a + c*width + b*width*depth];
				if (val >= 1.0e35)
				{
					val = 0.0;
				}
				val = val/max;

				get(i,j,k,0) = 255*val;
				get(i,j,k,1) = 255*val;
				get(i,j,k,2) = 255*val;
				get(i,j,k,3) = 255;
			}
		}

	}

}

DataTexture::~DataTexture() {
	// TODO Auto-generated destructor stub
}

DataTexture::DataTexture(int width, int height, float* data) {

	create(width, height, 1);

	float min = 1.0e35;
	float max = 0.0;

	for (int f = 0; f < width*height; f++)
	{
		float val = data[f];
		if (val < 1.0e35)
		{
			if (val < min) { min = val; }
			if (val > max) { max = val; }
		}
	}

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			float val = 255.0*(data[i + j*width]-min)/(max-min);
			get(i,j,0) = val > 0.0f && val < 1.0f ? 1.0f : val;
			get(i,j,1) = 0.0f;
			get(i,j,2) = 0.0f;
			get(i,j,3) = 0.0f;
		}

	}

	std::cout << "Max: " << max << " Min: " << min << std::endl;
}


}
