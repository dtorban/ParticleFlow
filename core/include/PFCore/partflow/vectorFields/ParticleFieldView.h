/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef PARTICLEFIELDVIEW_H_
#define PARTICLEFIELDVIEW_H_

#include "PFCore/math/v4.h"
#include <string>

namespace PFCore {
namespace partflow {

class ParticleFieldView {
public:
	PF_ENV_API ParticleFieldView();
	PF_ENV_API ParticleFieldView(const ParticleFieldView& particleField);
	PF_ENV_API ~ParticleFieldView() {}
	PF_ENV_API inline void operator=(const ParticleFieldView& particleField);

	PF_ENV_API inline int getNumValues() const;
	PF_ENV_API inline int getNumVectors() const;
	PF_ENV_API inline const math::vec4& getLength() const;
	PF_ENV_API inline const math::vec4& getStart() const;
	PF_ENV_API inline const math::vec4& getSize() const;
	PF_ENV_API inline int getDeviceId() const;
	PF_ENV_API inline const math::vec4 getIncrement() const;
	PF_ENV_API inline float* getValues(int valueIndex = 0) const;
	PF_ENV_API inline math::vec3* getVectors(int valueIndex = 0) const;

	size_t getSize() {
		return _size.x*_size.y*_size.z*_size.t*(_numValues*sizeof(float) + _numVectors*sizeof(math::vec3));
	}

protected:
	float* _values;
	math::vec3* _vectors;
	math::vec4 _start;
	math::vec4 _length;
	math::vec4 _size;
	int _numValues;
	int _numVectors;
	int _deviceId;
};

PF_ENV_API inline ParticleFieldView::ParticleFieldView() : _values(0), _vectors(0), _start(0), _length(0), _size(0), _numValues(0), _numVectors(0), _deviceId(-1)
{
}

PF_ENV_API inline ParticleFieldView::ParticleFieldView(const ParticleFieldView& particleField)
{
	*(this) = particleField;
}

PF_ENV_API inline void ParticleFieldView::operator=(const ParticleFieldView& particleField)
{
	_values = particleField._values;
	_vectors = particleField._vectors;
	_start = particleField._start;
	_length = particleField._length;
	_size = particleField._size;
	_numValues = particleField._numValues;
	_numVectors = particleField._numVectors;
	_deviceId = particleField._deviceId;
}

PF_ENV_API inline int ParticleFieldView::getNumValues() const {
	return _numValues;
}

PF_ENV_API inline int ParticleFieldView::getNumVectors() const {
	return _numVectors;
}

PF_ENV_API inline const math::vec4& ParticleFieldView::getLength() const {
	return _length;
}

PF_ENV_API inline const math::vec4& ParticleFieldView::getStart() const {
	return _start;
}

PF_ENV_API inline const math::vec4& ParticleFieldView::getSize() const {
	return _size;
}

PF_ENV_API inline int ParticleFieldView::getDeviceId() const {
	return _deviceId;
}

PF_ENV_API inline float* ParticleFieldView::getValues(int valueIndex) const {
	return &_values[(int)(_size.x*_size.y*_size.z*_size.t*valueIndex)];
}

PF_ENV_API inline math::vec3* ParticleFieldView::getVectors(int valueIndex) const {
	return &_vectors[(int)(_size.x*_size.y*_size.z*_size.t*valueIndex)];
}

PF_ENV_API inline const math::vec4 ParticleFieldView::getIncrement() const {
	return _length/(_size + 1);
}

} /* namespace partflow */
} /* namespace PFCore */

#endif /* PARTICLEFIELDVIEW_H_ */
