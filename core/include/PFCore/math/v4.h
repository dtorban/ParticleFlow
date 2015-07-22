/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef V4_H_
#define V4_H_

#include "PFCore/math/v3.h"

namespace PFCore {
namespace math {

template <typename T>
class v4 : public v3<T> {
public:
	union {T w, t;};

	PF_ENV_API v4();
	PF_ENV_API v4(const v3<T> &v, T wval);
	PF_ENV_API v4(T xval, T yval, T zval, T wval);
	PF_ENV_API v4(const v4<T> &v);
	PF_ENV_API v4(T a);
	PF_ENV_API ~v4();

	PF_ENV_API inline void operator=(const v4<T> &v);
	PF_ENV_API inline void operator+=(const v4<T> &v);
	PF_ENV_API inline v4 operator+(const v4<T> &v) const;
	PF_ENV_API inline void operator-=(const v4<T> &v);
	PF_ENV_API inline v4 operator-(const v4<T> &v) const;
	PF_ENV_API inline void operator*=(const v4<T> &v);
	PF_ENV_API inline v4 operator*(const v4<T> &v) const;
	PF_ENV_API inline void operator/=(const v4<T> &v);
	PF_ENV_API inline v4 operator/(const v4<T> &v) const;
};

template <typename T>
PF_ENV_API v4<T>::v4() {
}

template <typename T>
PF_ENV_API v4<T>::v4(const v3<T> &v, T wval) : v3<T>(v) {
	w = wval;
}

template <typename T>
PF_ENV_API v4<T>::v4(T xval, T yval, T zval, T wval) : v3<T>(xval, yval, zval) {
	w = wval;
}

template <typename T>
PF_ENV_API v4<T>::v4(const v4<T> &v) : v3<T>(v) {
	w = v.w;
}

template <typename T>
PF_ENV_API v4<T>::v4(T a) : v3<T>(a) {
	w = a;
}

template <typename T>
PF_ENV_API v4<T>::~v4() {
}

template <typename T>
PF_ENV_API void v4<T>::operator=(const v4<T> &v) {
	v3<T>::operator=(v);
	w = v.w;
}

template <typename T>
PF_ENV_API void v4<T>::operator+=(const v4<T> &v) {
	v3<T>::operator+=(v);
	w += v.w;
}

template <typename T>
PF_ENV_API v4<T> v4<T>::operator+(const v4<T> &v) const {
	v4<T> nv = *this;
	nv += v;
	return nv;
}

template <typename T>
PF_ENV_API void v4<T>::operator-=(const v4<T> &v) {
	v3<T>::operator-=(v);
	w -= v.w;
}

template <typename T>
PF_ENV_API v4<T> v4<T>::operator-(const v4<T> &v) const {
	v4<T> nv = *this;
	nv -= v;
	return nv;
}

template <typename T>
PF_ENV_API void v4<T>::operator*=(const v4<T> &v) {
	v3<T>::operator*=(v);
	w *= v.w;
}

template <typename T>
PF_ENV_API v4<T> v4<T>::operator*(const v4<T> &v) const {
	v4<T> nv = *this;
	nv *= v;
	return nv;
}

template <typename T>
PF_ENV_API void v4<T>::operator/=(const v4<T> &v) {
	v3<T>::operator/=(v);
	w /= v.w;
}

template <typename T>
PF_ENV_API v4<T> v4<T>::operator/(const v4<T> &v) const {
	v4<T> nv = *this;
	nv /= v;
	return nv;
}

typedef v4<int> ivec4;
typedef v4<float> vec4;
typedef v4<double> dvec4;

} /* namespace math */
}

#endif /* V3_H_ */
