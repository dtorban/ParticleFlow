/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/lgpl-3.0.html.
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef VEC_MATH_H_
#define VEC_MATH_H_

#include "math.h"
#include "math/v3.h"
#include "math/v4.h"

namespace PFCore {
namespace math {

PF_ENV_API inline vec3 ceil(const vec3 &v)
{
	return vec3(ceilf(v.x), ceilf(v.y), ceilf(v.z));
}

PF_ENV_API inline  vec4 ceil(const vec4 &v)
{
	return vec4(ceilf(v.x), ceilf(v.y), ceilf(v.z), ceilf(v.w));
}

PF_ENV_API inline  vec3 floor(const vec3 &v)
{
	return vec3(floorf(v.x), floorf(v.y), floorf(v.z));
}

PF_ENV_API inline  vec4 floor(const vec4 &v)
{
	return vec4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}

PF_ENV_API inline float length(const vec3 &v)
{
	return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

PF_ENV_API inline float length(const vec4 &v)
{
	return sqrt(v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w);
}

} /* namespace math */
}

#endif /* VEC_MATH_H_ */
