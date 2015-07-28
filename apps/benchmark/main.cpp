/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <iostream>
#include "PFCore/math/v3.h"
#include "PFCore/input/loaders/BlankLoader.h"
#include "PFCore/partflow/ParticleSet.h"
#include "PFCore/partflow/emitters/BasicEmitter.h"
#include "PFCore/partflow/emitters/strategies/SphereEmitter.h"
//#include "PFGpu/partflow/CudaParticleSetFactory.h"

using namespace PFCore::math;
using namespace PFCore::input;
using namespace PFCore::partflow;
using namespace std;

int main(int argc, char** argv) {

	//CudaParticleSetFactory f(0);
	//f.createParticleSet(10);


	BlankLoader loader(2.0);
	float data[10];
	loader.load(data,10);
	cout << data[5] << endl;

	vec3 test(1.0);
	cout << test.x << "," << test.y << "," << test.z << endl;

	ParticleSet pset(10, 0, 0, 3);
	for (int f = 0; f < pset.getNumParticles(); f++)
	{
		pset.getPosition(f) = vec3(0.0);
	}

	for (int f = 0; f < pset.getNumParticles(); f++)
	{
		pset.getPosition(f) += vec3(f);
	}

	for (int f = 0; f < pset.getNumParticles(); f++)
	{
		vec3& pos = pset.getPosition(f);
		cout << pos.x << "," << pos.y << "," << pos.z << endl;
	}

	cout << pset.getSize() << endl;

	BasicEmitter<SphereEmitter> emitter(SphereEmitter(vec3(0.0f), 1.0f, 1, RandomValue()));
	for (int f = 0; f < pset.getNumSteps(); f++)
	{
		emitter.emitParticles(pset, f);
	}

	for (int f = 0; f < pset.getNumParticles(); f++)
	{
		vec3& pos = pset.getPosition(f);
		cout << pos.x << "," << pos.y << "," << pos.z << endl;
	}

	return 0;
}
