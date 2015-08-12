/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef PARTICLESET_H_
#define PARTICLESET_H_

#include "PFCore/partflow/ParticleSetView.h"
#include "string.h"

namespace PFCore {
namespace partflow {

class ParticleSet : public ParticleSetView {
public:
	ParticleSet(int numParticles, int numAttributes, int numValues, int numVectors, int numSteps = 1) : ParticleSetView()
	{
		_numParticles = numParticles;
		_numAttributes = numAttributes;
		_numValues = numValues;
		_numVectors = numVectors;
		_numSteps = numSteps;
		_createdArrays = true;
		_positions = new math::vec3[numSteps*numParticles];
		_attributes = new int[numSteps*numParticles*numAttributes];
		_values = new float[numSteps*numParticles*numValues];
		_vectors = new math::vec3[numSteps*numParticles*numVectors];
	}

	virtual ~ParticleSet()
	{
		if (_createdArrays)
		{
			delete[] _positions;
			delete[] _attributes;
			delete[] _values;
			delete[] _vectors;
		}
	}

	void copy(const ParticleSetView& particleSet)
	{
		int numSteps = getNumSteps() < particleSet.getNumSteps() ? getNumSteps() : particleSet.getNumSteps();
		int numParticles = getNumParticles() < particleSet.getNumParticles() ? getNumParticles() : particleSet.getNumParticles();
		int numAttributes = getNumAttributes() < particleSet.getNumAttributes() ? getNumAttributes() : particleSet.getNumAttributes();
		int numVals = getNumValues() < particleSet.getNumValues() ? getNumValues() : particleSet.getNumValues();
		int numVectors = getNumVectors() < particleSet.getNumVectors() ? getNumVectors() : particleSet.getNumVectors();

		if (getNumParticles() == particleSet.getNumParticles())
		{
			int localStep = getStartStep(particleSet);

			copy(particleSet, (void*)getPositions(localStep), (void*)particleSet.getPositions(), numParticles*numSteps*sizeof(math::vec3));
			copy(particleSet, (void*)getAttributes(0, localStep), (void*)particleSet.getAttributes(), numParticles*numSteps*numAttributes*sizeof(float));
			copy(particleSet, (void*)getValues(0, localStep), (void*)particleSet.getValues(), numParticles*numSteps*numVals*sizeof(float));
			copy(particleSet, (void*)getVectors(0, localStep), (void*)particleSet.getVectors(), numParticles*numSteps*numVectors*sizeof(math::vec3));
		}
		else
		{
			for (int step = 0; step < numSteps; step++)
			{
				int localStep = step + getStartStep(particleSet);

				for (int index = 0; index < numParticles; index++)
				{
					int localIndex = getStartIndex(particleSet);

					copy(particleSet, (void*)(getPositions(localStep) + localIndex), (void*)particleSet.getPositions(step), numParticles*sizeof(math::vec3));

					for (int valIndex = 0; valIndex < numVals; valIndex++)
					{
						copy(particleSet, (void*)(getAttributes(valIndex, localStep) + localIndex), (void*)particleSet.getAttributes(valIndex, step), numParticles*sizeof(float));
					}

					for (int valIndex = 0; valIndex < numVals; valIndex++)
					{
						copy(particleSet, (void*)(getValues(valIndex, localStep) + localIndex), (void*)particleSet.getValues(valIndex, step), numParticles*sizeof(float));
					}

					for (int valIndex = 0; valIndex < numVectors; valIndex++)
					{
						copy(particleSet, (void*)(getVectors(valIndex, localStep) + localIndex), (void*)particleSet.getVectors(valIndex, step), numParticles*sizeof(math::vec3));
					}
				}
			}
		}
	}

private:
	bool _createdArrays;

protected:
	ParticleSet() : ParticleSetView(), _createdArrays(false)
	{
	}

	virtual void copy(const ParticleSetView& particleSet, void* dst, const void* src, size_t size)
	{
		memcpy(dst, src, size);
	}
};

} /* namespace partflow */
} /* namespace PFCore */

#endif /* PARTICLESET_H_ */
