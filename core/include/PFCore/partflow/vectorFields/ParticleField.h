/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef PARTICLEFIELD_H_
#define PARTICLEFIELD_H_

#include "PFCore/partflow/vectorFields/ParticleFieldView.h"

namespace PFCore {
namespace partflow {

class ParticleField : public ParticleFieldView {
public:
	ParticleField(int numValues, int numVectors, math::vec4 start, math::vec4 length, math::vec4 size);
	virtual ~ParticleField()
	{
		if (_createdArrays)
		{
			delete[] _values;
			delete[] _vectors;
		}
	}

	void copy(const ParticleFieldView& particleField)
	{
		copy(particleField, _values, particleField.getValues(0), _size.x*_size.y*_size.z*_size.t*_numValues*sizeof(float));
		copy(particleField, _vectors, particleField.getValues(0), _size.x*_size.y*_size.z*_size.t*_numVectors*sizeof(math::vec3));
	}

protected:
	ParticleField();
	virtual void copy(const ParticleFieldView& particleField, void* dst, const void* src, size_t size)
	{
		memcpy(dst, src, size);
	}

private:
	bool _createdArrays;
};

inline ParticleField::ParticleField(int numValues, int numVectors, math::vec4 start, math::vec4 length, math::vec4 size) : ParticleFieldView()
{
	_createdArrays = true;
	_deviceId = -1;
	_numValues = numValues;
	_numVectors = numVectors;
	_start = start;
	_length = length;
	_size = size;
	_values = new float[(int)(size.x*size.y*size.z*size.t*numValues)];
	_vectors = new math::vec3[(int)(size.x*size.y*size.z*size.t*numVectors)];
}

inline ParticleField::ParticleField() : ParticleFieldView(), _createdArrays(false)
{
}

} /* namespace partflow */
} /* namespace PFCore */

#endif /* PARTICLEFIELD_H_ */
