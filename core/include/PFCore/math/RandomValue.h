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
namespace math {

struct RandomValueGenerator
{
	PF_ENV_API inline float getValue(int i) const
	{
		return float(std::rand())/RAND_MAX;
	}

	PF_ENV_API inline void operator=(const RandomValueGenerator &rndVal)
	{
	}

	PF_ENV_API inline void randomize(int seed)
	{
	}
};

struct RandomArrayValue
{
	PF_ENV_API RandomArrayValue() : rnd(0), numRand(0), frameRand(0) {}
	PF_ENV_API RandomArrayValue(const RandomArrayValue& rndVal)
	{
		rnd = rndVal.rnd;
		numRand = rndVal.numRand;
		frameRand = rndVal.frameRand;
	}

	float* rnd;
	int numRand;
	int frameRand;

	PF_ENV_API inline float getValue(int i) const
	{
		return rnd[(i + frameRand)%numRand];
	}

	PF_ENV_API inline void operator=(const RandomArrayValue &rndVal)
	{
		rnd = rndVal.rnd;
		numRand = rndVal.numRand;
		frameRand = rndVal.frameRand;
	}

	PF_ENV_API inline void randomize(int seed)
	{
		frameRand = numRand*float(std::rand())/RAND_MAX;
	}
};

#ifndef RandomValue
#define RandomValue RandomValueGenerator
#endif

} /* namespace partflow */
} /* namespace PFCore */

#endif /* RANDOMVALUE_H_ */
