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
#include "PFGpu/partflow/GpuParticleSetFactory.h"
#include "PFGpu/partflow/emitters/GpuEmitterFactory.h"
#include "PFCore/partflow/PartflowRef.h"

using namespace PFCore::math;
using namespace PFCore::input;
using namespace PFCore::partflow;
using namespace std;

void printParticleSet(const ParticleSetView& view);

int main(int argc, char** argv) {

	GpuParticleSetFactory psetFactory;
	ParticleSetRef localSet = psetFactory.createLocalParticleSet(10);
	ParticleSetRef updatedSet = psetFactory.createLocalParticleSet(10);
	ParticleSetRef deviceSet = psetFactory.createParticleSet(0, 10);

	GpuEmitterFactory emitterFactory;
	EmitterRef emitter = EmitterRef(emitterFactory.createSphereEmitter(vec3(0.0f), 1.0f, 1));

	//BasicEmitter<SphereEmitter> emitter(SphereEmitter(vec3(0.0f), 1.0f, 1, RandomValue()));
	for (int f = 0; f < localSet->getNumSteps(); f++)
	{
		emitter->emitParticles(*localSet, f);
	}

	// Print out local
	cout << "Local: " << endl;
	printParticleSet(*localSet);
	//printParticleSet(*deviceSet);

	// Copy to device
	deviceSet->copy(localSet->getView().filterBySize(1,5));
	deviceSet->copy(localSet->getView().filterBySize(7,2));

	// Copy from device
	updatedSet->copy(*deviceSet);

	// Print out device
	cout << "Device: " << endl;
	printParticleSet(*updatedSet);

	EmitterRef emitterGpu = EmitterRef(emitterFactory.createSphereEmitter(vec3(0.0f), 2.0f, 1));
	for (int f = 0; f < deviceSet->getNumSteps(); f++)
	{
		emitterGpu->emitParticles(*deviceSet, f);
	}

	// Copy from device
	updatedSet->copy(*deviceSet);

	// Print out device
	cout << "Device new: " << endl;
	printParticleSet(*updatedSet);

	return 0;
}

void printParticleSet(const ParticleSetView& view)
{
	for (int f = 0; f < view.getNumParticles(); f++)
	{
		const vec3& pos = view.getPosition(f);
		cout << pos.x << "," << pos.y << "," << pos.z << endl;
	}
}
