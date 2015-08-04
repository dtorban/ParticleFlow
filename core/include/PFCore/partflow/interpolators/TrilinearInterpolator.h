/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef TRILINEARINTERPOLATOR_H_
#define TRILINEARINTERPOLATOR_H_

#include "PFCore/math/v4.h"
#include "PFCore/math/vec_math.h"

namespace PFCore {
namespace partflow {

template <typename T, typename S>
class TrilinearInterpolator {
public:
	PF_ENV_API TrilinearInterpolator() {}
	PF_ENV_API ~TrilinearInterpolator() {}

	PF_ENV_API T interpolate(T* values, const math::v4<S> &pos, const math::v4<S> &start, const math::v4<S> &length, const math::v4<S> &size);

private:
	PF_ENV_API T& getValue(T* values, const math::v4<S> &size, int x, int y, int z, int t);
};

template <typename T, typename S>
PF_ENV_API T& TrilinearInterpolator<T, S>::getValue(T* values, const math::v4<S> &size, int x, int y, int z, int t)
{
	return values[(int)(x + y*size.x + z*size.x*size.y + (t % (int)size.t)*size.x*size.y*size.z)];
}

template <typename T, typename S>
PF_ENV_API T TrilinearInterpolator<T, S>::interpolate(T* values, const math::v4<S> &pos, const math::v4<S> &start, const math::v4<S> &length, const math::v4<S> &size)
{
	math::v4<S> v, v0, v1, d;
	math::v4<S> len;
	len.x = length.x == 0.0 ? 1.0 : length.x;
	len.y = length.y == 0.0 ? 1.0 : length.y;
	len.z = length.z == 0.0 ? 1.0 : length.z;
	len.t = length.t == 0.0 ? 1.0 : length.t;

	math::v4<S> p = (pos - start)/len;
	v = p*(size-1);

	v0 = math::floor(v);
	v1 = math::ceil(v);
	d = v - v0;


	if (p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1 || p.z < 0 || p.z > 1)
	{
		return 0.0;
	}

	T c[16];
	c[0] = getValue(values, size, v0.x, v0.y, v0.z, v0.t);
	c[1] = getValue(values, size, v1.x, v0.y, v0.z, v0.t);
	c[2] = getValue(values, size, v0.x, v0.y, v1.z, v0.t);
	c[3] = getValue(values, size, v1.x, v0.y, v1.z, v0.t);
	c[4] = getValue(values, size, v0.x, v1.y, v0.z, v0.t);
	c[5] = getValue(values, size, v1.x, v1.y, v0.z, v0.t);
	c[6] = getValue(values, size, v0.x, v1.y, v1.z, v0.t);
	c[7] = getValue(values, size, v1.x, v1.y, v1.z, v0.t);
	c[8] = getValue(values, size, v0.x, v0.y, v0.z, v1.t);
	c[9] = getValue(values, size, v1.x, v0.y, v0.z, v1.t);
	c[10] = getValue(values, size, v0.x, v0.y, v1.z, v1.t);
	c[11] = getValue(values, size, v1.x, v0.y, v1.z, v1.t);
	c[12] = getValue(values, size, v0.x, v1.y, v0.z, v1.t);
	c[13] = getValue(values, size, v1.x, v1.y, v0.z, v1.t);
	c[14] = getValue(values, size, v0.x, v1.y, v1.z, v1.t);
	c[15] = getValue(values, size, v1.x, v1.y, v1.z, v1.t);

	T t[8];
	for (int f = 0; f < 8; f++)
	{
		t[f] = c[f] + ((c[f+8]-c[f]) * d.t);
	}

	T x[4];
	for (int f = 0; f < 4; f++)
	{
		x[f] = t[f*2] + ((t[f*2+1]-t[f*2]) * d.x);
	}

	T y[2];
	for (int f = 0; f < 2; f++)
	{
		y[f] = x[f] + ((x[f+2]-x[f]) * d.y);
	}

	return y[0] + ((y[1]-y[0]) * d.z);
}

} /* namespace partflow */
} /* namespace PFCore */

#endif /* TRILINEARINTERPOLATOR_H_ */
