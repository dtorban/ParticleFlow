/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef PARTICLEFIELDVOLUME_H_
#define PARTICLEFIELDVOLUME_H_

#include "PFCore/partflow/vectorFields/ParticleFieldView.h"
#include "PFCore/partflow/interpolators/TrilinearInterpolator.h"

namespace PFCore {
namespace partflow {

class ParticleFieldVolume {
public:
	PF_ENV_API ParticleFieldVolume() {}
	PF_ENV_API ParticleFieldVolume(const ParticleFieldVolume& volume) { *(this) = volume; }
	PF_ENV_API ParticleFieldVolume(const ParticleFieldView& particleField, int velocityVectorIndex = 0);
	PF_ENV_API virtual ~ParticleFieldVolume() {}
	PF_ENV_API inline void operator=(const ParticleFieldVolume& volume);

	PF_ENV_API inline const ParticleFieldView& getParticleField() const;
	PF_ENV_API inline math::vec3 getVector(int valueIndex, const math::vec3& pos, float time);
	PF_ENV_API inline float getValue(int valueIndex, const math::vec3& pos, float time);
	PF_ENV_API inline math::vec3 getVelocity(const math::vec3& pos, float time);

	std::string getTypeId() { return "ParticleFieldVolume"; }

private:
	ParticleFieldView _particleField;
	int _velocityVectorIndex;
	TrilinearInterpolator<math::vec3, float> _vectorInterp;
	TrilinearInterpolator<float, float> _valueInterp;
};

PF_ENV_API inline ParticleFieldVolume::ParticleFieldVolume(const ParticleFieldView& particleField, int velocityVectorIndex) : _particleField(particleField), _velocityVectorIndex(velocityVectorIndex)
{
}

PF_ENV_API inline void ParticleFieldVolume::operator=(const ParticleFieldVolume& volume)
{
	_particleField = volume._particleField;
	_velocityVectorIndex = volume._velocityVectorIndex;
}

PF_ENV_API inline const ParticleFieldView& ParticleFieldVolume::getParticleField() const
{
	return _particleField;
}

PF_ENV_API inline math::vec3 ParticleFieldVolume::getVector(int valueIndex, const math::vec3& pos, float time)
{
	return _vectorInterp.interpolate(_particleField.getVectors(valueIndex),
			math::vec4(pos, time),
			_particleField.getStart(),
			_particleField.getLength(),
			_particleField.getSize());
}

PF_ENV_API inline float ParticleFieldVolume::getValue(int valueIndex, const math::vec3& pos, float time)
{
	return _valueInterp.interpolate(_particleField.getValues(valueIndex),
			math::vec4(pos, time),
			_particleField.getStart(),
			_particleField.getLength(),
			_particleField.getSize());
}

PF_ENV_API inline math::vec3 ParticleFieldVolume::getVelocity(const math::vec3& pos, float time)
{
	return getVector(_velocityVectorIndex, pos, time);
}

} /* namespace partflow */
} /* namespace PFCore */

#endif /* PARTICLEFIELDVOLUME_H_ */
