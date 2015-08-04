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
	PF_ENV_API ParticleSetView(const ParticleSetView& particleSet)
	{
		_positions = particleSet._positions;
		_values = particleSet._values;
		_vectors = particleSet._vectors;
		_numParticles = particleSet._numParticles;
		_numValues = particleSet._numValues;
		_numVectors = particleSet._numVectors;
		_numSteps = particleSet._numSteps;
		_startIndex = particleSet._startIndex;
		_length = particleSet._length;
		_startStep = particleSet._startStep;
		_stepLength = particleSet._stepLength;
		_deviceId = particleSet._deviceId;
	}

	PF_ENV_API ~ParticleSetView() {}

	PF_ENV_API inline int getNumParticles() const;
	PF_ENV_API inline int getNumValues() const;
	PF_ENV_API inline int getNumVectors() const;
	PF_ENV_API inline int getNumSteps() const;
	PF_ENV_API inline int getDeviceId() const;

	PF_ENV_API inline const math::vec3* getPositions(int step = 0) const;
	PF_ENV_API inline float* getValues(int valueIndex = 0, int step = 0) const;
	PF_ENV_API inline const math::vec3* getVectors(int valueIndex = 0, int step = 0) const;

	PF_ENV_API inline math::vec3& getPosition(int index, int step = 0);
	PF_ENV_API inline math::vec3& getPosition(int index, int step = 0) const;
	PF_ENV_API inline float& getValue(int valueIndex, int index, int step = 0);
	PF_ENV_API inline float& getValue(int valueIndex, int index, int step = 0) const;
	PF_ENV_API inline math::vec3& getVector(int valueIndex, int index, int step = 0);
	PF_ENV_API inline math::vec3& getVector(int valueIndex, int index, int step = 0) const;

	PF_ENV_API inline void operator=(const ParticleSetView& particleSet);

	PF_ENV_API inline ParticleSetView getView();
	PF_ENV_API inline ParticleSetView& filterBySize(int startIndex, int length);
	PF_ENV_API inline ParticleSetView& filterByStep(int startStep, int length);
	PF_ENV_API inline ParticleSetView& filterReset();

	size_t getSize() {
		return getStepLength()*getLength()*(sizeof(math::vec3) + _numValues*sizeof(float) + _numVectors*sizeof(math::vec3));
	}

protected:
	PF_ENV_API ParticleSetView() :
		_positions(0), _values(0), _vectors(0), _numParticles(0), _numValues(0), _numVectors(
				0), _numSteps(0), _startIndex(0), _length(-1), _startStep(0), _stepLength(-1), _deviceId(-1)
	{
	}

	PF_ENV_API inline int getLength() const;
	PF_ENV_API inline int getStepLength() const;
	PF_ENV_API inline int getStartStep(const ParticleSetView& particleSet) const;
	PF_ENV_API inline int getStartIndex(const ParticleSetView& particleSet) const;
	PF_ENV_API inline int calcStep(int step) const;

	math::vec3* _positions;
	float* _values;
	math::vec3* _vectors;
	int _numParticles;
	int _numValues;
	int _numVectors;
	int _numSteps;
	int _deviceId;
	int _startIndex;
	int _length;
	int _startStep;
	int _stepLength;
};

PF_ENV_API inline int ParticleSetView::getNumParticles() const {
	return getLength();
}

PF_ENV_API inline int ParticleSetView::getNumValues() const {
	return _numValues;
}
PF_ENV_API inline int ParticleSetView::getNumVectors() const {
	return _numVectors;
}

PF_ENV_API inline int ParticleSetView::getNumSteps() const {
	return getStepLength();
}

PF_ENV_API inline int ParticleSetView::getDeviceId() const {
	return _deviceId;
}

PF_ENV_API inline const math::vec3* ParticleSetView::getPositions(int step) const {
	return &_positions[_numParticles*calcStep(step) + _startIndex];
}

PF_ENV_API inline float* ParticleSetView::getValues(int valueIndex, int step) const {
	return &_values[_numParticles*_numValues*calcStep(step) + valueIndex*_numParticles + _startIndex];
}

PF_ENV_API inline const math::vec3* ParticleSetView::getVectors(int valueIndex, int step) const {
	return &_vectors[_numParticles*_numVectors*calcStep(step) + valueIndex*_numParticles + _startIndex];
}

PF_ENV_API inline math::vec3& ParticleSetView::getPosition(int index, int step) {
	return _positions[calcStep(step)*_numParticles + _startIndex + index];
}

PF_ENV_API inline math::vec3& ParticleSetView::getPosition(int index, int step) const {
	return _positions[calcStep(step)*_numParticles + _startIndex + index];
}

PF_ENV_API inline float& ParticleSetView::getValue(int valueIndex, int index, int step) {
	return _values[calcStep(step)*_numParticles*_numValues + valueIndex*_numParticles + _startIndex + index];
}

PF_ENV_API inline float& ParticleSetView::getValue(int valueIndex, int index, int step) const {
	return _values[calcStep(step)*_numParticles*_numValues + valueIndex*_numParticles + _startIndex + index];
}

PF_ENV_API inline math::vec3& ParticleSetView::getVector(int valueIndex, int index, int step) {
	return _vectors[calcStep(step)*_numParticles*_numVectors + valueIndex*_numParticles + _startIndex + index];
}

PF_ENV_API inline math::vec3& ParticleSetView::getVector(int valueIndex, int index, int step) const {
	return _vectors[calcStep(step)*_numParticles*_numVectors + valueIndex*_numParticles + _startIndex + index];
}

PF_ENV_API inline void ParticleSetView::operator =(const ParticleSetView& particleSet) {
	_positions = particleSet._positions;
	_values = particleSet._values;
	_vectors = particleSet._vectors;
	_numParticles = particleSet._numParticles;
	_numValues = particleSet._numValues;
	_numVectors = particleSet._numVectors;
	_numSteps = particleSet._numSteps;
	_startIndex = particleSet._startIndex;
	_length = particleSet._length;
	_startStep = particleSet._startStep;
	_stepLength = particleSet._stepLength;
	_deviceId = particleSet._deviceId;
}

PF_ENV_API inline ParticleSetView& ParticleSetView::filterBySize(int startIndex, int length) {
	_startIndex += startIndex;
	int len = getLength() - startIndex;
	_length = length < len ? length : len;
	return *this;
}

PF_ENV_API inline ParticleSetView& ParticleSetView::filterByStep(int startStep, int length) {
	_startStep += startStep;
	int len = getStepLength() - startStep;
	_stepLength = length < len ? length : len;
	return *this;
}

PF_ENV_API inline ParticleSetView& ParticleSetView::filterReset() {
	_startIndex = 0;
	_length = -1;
	_startStep = 0;
	_stepLength = -1;
	return *this;
}

PF_ENV_API inline int ParticleSetView::getLength() const
{
	return _length > 0 ? _length : _numParticles;
}

PF_ENV_API inline int ParticleSetView::getStepLength() const
{
	return _stepLength > 0 ? _stepLength : _numSteps;
}

PF_ENV_API inline int ParticleSetView::getStartStep(const ParticleSetView& particleSet) const
{
	return particleSet._startStep;
}

PF_ENV_API inline int ParticleSetView::getStartIndex(const ParticleSetView& particleSet) const
{
	return particleSet._startIndex;
}

PF_ENV_API inline int ParticleSetView::calcStep(int step) const
{
	return (step + _startStep) % _numSteps;
}

PF_ENV_API inline ParticleSetView ParticleSetView::getView() {
	return ParticleSetView(*this);
}

} /* namespace partflow */
}

#endif /* PARTICLESETVIEW_H_ */
