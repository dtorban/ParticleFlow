/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef PARTICLESETVIEW_H_
#define PARTICLESETVIEW_H_

#include "PFCore/math/v3.h"

namespace PFCore {
namespace partflow {

class ParticleSetView {
public:
	PF_ENV_API ParticleSetView(const ParticleSetView& particleSet);
	PF_ENV_API ~ParticleSetView();

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

protected:
	math::vec3* _positions;
	float* _values;
	math::vec3* _vectors;
	int _numParticles;
	int _numValues;
	int _numVectors;
	int _numSteps;
	PF_ENV_API ParticleSetView();
};

PF_ENV_API ParticleSetView::ParticleSetView() :
		_positions(0), _values(0), _vectors(0), _numParticles(0), _numValues(0), _numVectors(
				0), _numSteps(0)
{
}

PF_ENV_API ParticleSetView::~ParticleSetView()
{
}

PF_ENV_API ParticleSetView::ParticleSetView(const ParticleSetView& particleSet)
{
	_positions = particleSet._positions;
	_values = particleSet._values;
	_vectors = particleSet._vectors;
	_numParticles = particleSet._numParticles;
	_numValues = particleSet._numValues;
	_numVectors = particleSet._numVectors;
	_numSteps = particleSet._numSteps;
}

PF_ENV_API inline int ParticleSetView::getNumParticles() const {
	return _numParticles;
}

PF_ENV_API inline int ParticleSetView::getNumValues() const {
	return _numValues;
}
PF_ENV_API inline int ParticleSetView::getNumVectors() const {
	return _numVectors;
}

PF_ENV_API inline int ParticleSetView::getNumSteps() const {
	return _numSteps;
}

PF_ENV_API inline const math::vec3* ParticleSetView::getPositions(int step) const {
	return &_positions[_numParticles*step];
}

PF_ENV_API inline float* ParticleSetView::getValues(int step) const {
	return &_values[_numParticles*_numValues*step];
}

PF_ENV_API inline const math::vec3* ParticleSetView::getVectors(int step) const {
	return &_vectors[_numParticles*_numVectors*step];
}

PF_ENV_API inline math::vec3& ParticleSetView::getPosition(int index, int step) {
	return _positions[step*_numParticles + index];
}

PF_ENV_API inline float& ParticleSetView::getValue(int valueIndex, int index, int step) {
	return _values[_numSteps*_numParticles*_numValues + valueIndex*_numParticles + index];
}

PF_ENV_API inline math::vec3& ParticleSetView::getVector(int valueIndex, int index, int step) {
	return _vectors[_numSteps*_numParticles*_numVectors + valueIndex*_numParticles + index];
}

inline size_t ParticleSetView::getSize() {
	return _numSteps*_numParticles*(sizeof(math::vec3) + _numValues*sizeof(float) + _numVectors*sizeof(math::vec3));
}

} /* namespace partflow */
}

#endif /* PARTICLESETVIEW_H_ */
