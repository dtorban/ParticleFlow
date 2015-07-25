/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef RANDOMVALUE_H_
#define RANDOMVALUE_H_

#include "PFCore/env.h"
#include <cstdlib>

namespace PFCore {
namespace partflow {

struct RandomValueGenerator
{
	PF_ENV_API inline float getValue(int i) const
	{
		return float(std::rand())/RAND_MAX;
	}
};

struct RandomArrayValue
{
	float* rnd;
	int numRand;
	int frameRand;

	PF_ENV_API inline float getValue(int i) const
	{
		return rnd[(i + frameRand)%numRand];
	}
};

#ifndef RandomValue
#define RandomValue RandomValueGenerator
#endif

} /* namespace partflow */
} /* namespace PFCore */

#endif /* RANDOMVALUE_H_ */
