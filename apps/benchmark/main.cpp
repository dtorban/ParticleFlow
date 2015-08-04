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
#include "PFCore/partflow/advectors/VectorFieldAdvector.h"
#include "PFGpu/partflow/advectors/GpuVectorFieldAdvector.h"
#include "PFCore/partflow/advectors/strategies/EulerAdvector.h"
#include "PFCore/partflow/vectorFields/ConstantField.h"

using namespace PFCore::math;
using namespace PFCore::input;
using namespace PFCore::partflow;
using namespace std;

void printParticleSet(const ParticleSetView& view, bool printVelocity = false);

int main(int argc, char** argv) {

	GpuParticleSetFactory psetFactory;
	ParticleSetRef localSet = psetFactory.createLocalParticleSet(10);
	ParticleSetRef updatedSet = psetFactory.createLocalParticleSet(10);
	ParticleSetRef deviceSet = psetFactory.createParticleSet(0, 10);

	GpuEmitterFactory emitterFactory;
	EmitterRef emitter = EmitterRef(emitterFactory.createSphereEmitter(vec3(0.0f), 1.0f, 1));

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

	// Advect device set
	AdvectorRef advector = AdvectorRef(new GpuVectorFieldAdvector<EulerAdvector<ConstantField>, ConstantField>(EulerAdvector<ConstantField>(), ConstantField(vec3(1.0f ,0.0f, 0.0f))));
	float dt = 0.1f;
	for (int f = 0; f < 10; f++)
	{
		advector->advectParticles(*deviceSet, f, dt*float(f), dt);
	}

	// Copy from device
	cout << "Device advected: " << endl;
	updatedSet->copy(*deviceSet);
	printParticleSet(*updatedSet, true);

	return 0;
}

void printParticleSet(const ParticleSetView& view, bool printVelocity)
{
	for (int f = 0; f < view.getNumParticles(); f++)
	{
		const vec3& pos = view.getPosition(f);
		cout << pos.x << "," << pos.y << "," << pos.z << endl;
	}

	if (printVelocity)
	{
		for (int f = 0; f < view.getNumParticles(); f++)
		{
			const vec3& vel = view.getVector(0,f,0);
			cout << "Vel: " << vel.x << "," << vel.y << "," << vel.z << endl;
		}
	}
}
