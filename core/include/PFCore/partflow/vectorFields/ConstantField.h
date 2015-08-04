/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef CONSTANTFIELD_H_
#define CONSTANTFIELD_H_

#include "PFCore/math/v3.h"
#include <string>

namespace PFCore {
namespace partflow {

class ConstantField {
public:
	PF_ENV_API ConstantField(const math::vec3& vec) : _vec(vec) {}
	PF_ENV_API ~ConstantField() {}

	PF_ENV_API inline math::vec3 getVelocity(const math::vec3& pos, float time);
	std::string getTypeId() { return "ConstantField"; }

private:
	math::vec3 _vec;
};

PF_ENV_API inline math::vec3 ConstantField::getVelocity(const math::vec3& pos, float time)
{
	return _vec;
}

} /* namespace partflow */
} /* namespace PFCore */

#endif /* CONSTANTFIELD_H_ */
