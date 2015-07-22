/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef V3_H_
#define V3_H_

#include "PFCore/env.h"

namespace PFCore {
namespace math {

template<typename T>
class v3;

typedef v3<int> ivec3;
typedef v3<float> vec3;
typedef v3<double> dvec3;

template<typename T>
class v3 {
public:
	T x, y, z;

	PF_ENV_API v3();
	PF_ENV_API v3(const v3<T> &v);
	PF_ENV_API v3(T a);
	PF_ENV_API v3(T xval, T yval, T zval);
	~v3();

	PF_ENV_API inline void operator=(const v3<T> &v);
	PF_ENV_API inline void operator+=(const v3<T> &v);
	PF_ENV_API inline v3 operator+(const v3<T> &v) const;
	PF_ENV_API inline void operator-=(const v3<T> &v);
	PF_ENV_API inline v3 operator-(const v3<T> &v) const;
	PF_ENV_API inline void operator*=(const v3<T> &v);
	PF_ENV_API inline v3 operator*(const v3<T> &v) const;
	PF_ENV_API inline void operator/=(const v3<T> &v);
	PF_ENV_API inline v3 operator/(const v3<T> &v) const;
};

template<typename T>
PF_ENV_API v3<T>::v3() {
}

template<typename T>
PF_ENV_API v3<T>::v3(T a) {
	x = a;
	y = a;
	z = a;
}

template<typename T>
PF_ENV_API v3<T>::v3(const v3<T> &v) {
	x = v.x;
	y = v.y;
	z = v.z;
}

template<typename T>
PF_ENV_API v3<T>::v3(T xval, T yval, T zval) {
	x = xval;
	y = yval;
	z = zval;
}

template<typename T>
PF_ENV_API v3<T>::~v3() {
}

template<typename T>
PF_ENV_API void v3<T>::operator=(const v3<T> &v) {
	x = v.x;
	y = v.y;
	z = v.z;
}

template<typename T>
PF_ENV_API void v3<T>::operator+=(const v3<T> &v) {
	x += v.x;
	y += v.y;
	z += v.z;
}

template<typename T>
PF_ENV_API v3<T> v3<T>::operator+(const v3<T> &v) const {
	v3<T> nv = *this;
	nv += v;
	return nv;
}

template<typename T>
PF_ENV_API void v3<T>::operator-=(const v3<T> &v) {
	x -= v.x;
	y -= v.y;
	z -= v.z;
}

template<typename T>
PF_ENV_API v3<T> v3<T>::operator-(const v3<T> &v) const {
	v3<T> nv = *this;
	nv -= v;
	return nv;
}

template<typename T>
PF_ENV_API void v3<T>::operator*=(const v3<T> &v) {
	x *= v.x;
	y *= v.y;
	z *= v.z;
}

template<typename T>
PF_ENV_API v3<T> v3<T>::operator*(const v3<T> &v) const {
	v3<T> nv = *this;
	nv *= v;
	return nv;
}

template<typename T>
PF_ENV_API void v3<T>::operator/=(const v3<T> &v) {
	x /= v.x;
	y /= v.y;
	z /= v.z;
}

template<typename T>
PF_ENV_API v3<T> v3<T>::operator/(const v3<T> &v) const {
	v3<T> nv = *this;
	nv /= v;
	return nv;
}

} /* namespace math */
}

#endif /* V3_H_ */
