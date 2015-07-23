/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef PARTICLESET_H_
#define PARTICLESET_H_

#include "PFCore/math/v3.h"

namespace PFCore {
namespace partflow {

class ParticleSet {
public:
	PF_ENV_API ParticleSet(const ParticleSet& particleSet);
	ParticleSet(int numParticles, int numValues, int numVectors, int numSteps = 1);
	virtual ~ParticleSet();

	PF_ENV_API inline int getNumParticles() const;
	PF_ENV_API inline int getNumValues() const;
	PF_ENV_API inline int getNumVectors() const;
	PF_ENV_API inline int getNumSteps() const;

	PF_ENV_API inline const math::vec3* getPositions(int step = 0) const;
	PF_ENV_API inline float* getValues(int step = 0) const;
	PF_ENV_API inline const math::vec3* getVectors(int step = 0) const;

	PF_ENV_API inline math::vec3& getPosition(int index, int step = 0);
	PF_ENV_API inline float& getValue(int valueIndex, int index, int step = 0);
	PF_ENV_API inline math::vec3& getVector(int valueIndex, int index, int step = 0);

	size_t getSize();

private:
	bool _createdArrays;

protected:
	math::vec3* _positions;
	float* _values;
	math::vec3* _vectors;
	int _numParticles;
	int _numValues;
	int _numVectors;
	int _numSteps;
	PF_ENV_API ParticleSet();
};

PF_ENV_API ParticleSet::ParticleSet() :
		_positions(0), _values(0), _vectors(0), _numParticles(0), _numValues(0), _numVectors(
				0), _numSteps(0), _createdArrays(false)
{
}

PF_ENV_API ParticleSet::ParticleSet(const ParticleSet& particleSet)
{
	_positions = particleSet._positions;
	_values = particleSet._values;
	_vectors = particleSet._vectors;
	_numParticles = particleSet._numParticles;
	_numValues = particleSet._numValues;
	_numVectors = particleSet._numVectors;
	_numSteps = particleSet._numSteps;
	_createdArrays = false;
}

ParticleSet::ParticleSet(int numParticles, int numValues, int numVectors, int numSteps) : _numParticles(numParticles), _numValues(numValues), _numVectors(numVectors), _numSteps(numSteps)
{
	_createdArrays = true;
	_positions = new math::vec3[numSteps*numParticles];
	_values = new float[numSteps*numParticles*numValues];
	_vectors = new math::vec3[numSteps*numParticles*numVectors];
}

ParticleSet::~ParticleSet()
{
	if (_createdArrays)
	{
		delete[] _positions;
		delete[] _values;
		delete[] _vectors;
	}
}

PF_ENV_API inline int ParticleSet::getNumParticles() const {
	return _numParticles;
}

PF_ENV_API inline int ParticleSet::getNumValues() const {
	return _numValues;
}
PF_ENV_API inline int ParticleSet::getNumVectors() const {
	return _numVectors;
}

PF_ENV_API inline int ParticleSet::getNumSteps() const {
	return _numSteps;
}

PF_ENV_API inline const math::vec3* ParticleSet::getPositions(int step) const {
	return &_positions[_numParticles*step];
}

PF_ENV_API inline float* ParticleSet::getValues(int step) const {
	return &_values[_numParticles*_numValues*step];
}

PF_ENV_API inline const math::vec3* ParticleSet::getVectors(int step) const {
	return &_vectors[_numParticles*_numVectors*step];
}

PF_ENV_API inline math::vec3& ParticleSet::getPosition(int index, int step) {
	return _positions[step*_numParticles + index];
}

PF_ENV_API inline float& ParticleSet::getValue(int valueIndex, int index, int step) {
	return _values[_numSteps*_numParticles*_numValues + valueIndex*_numParticles + index];
}

PF_ENV_API inline math::vec3& ParticleSet::getVector(int valueIndex, int index, int step) {
	return _vectors[_numSteps*_numParticles*_numVectors + valueIndex*_numParticles + index];
}

inline size_t ParticleSet::getSize() {
	return _numParticles*(sizeof(math::vec3) + _numValues*sizeof(float) + _numVectors*sizeof(math::vec3));
}

} /* namespace partflow */
}

#endif /* PARTICLESET_H_ */
