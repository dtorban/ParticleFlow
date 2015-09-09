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
#include "PFGpu/partflow/GpuParticleFactory.h"
#include "PFGpu/partflow/emitters/GpuEmitterFactory.h"
#include "PFCore/partflow/PartflowRef.h"
#include "PFCore/partflow/advectors/VectorFieldAdvector.h"
#include "PFGpu/partflow/advectors/GpuVectorFieldAdvector.h"
#include "PFCore/partflow/advectors/strategies/EulerAdvector.h"
#include "PFCore/partflow/advectors/strategies/RungaKutta4.h"
#include "PFCore/partflow/vectorFields/ConstantField.h"
#include "PFCore/partflow/updaters/BasicUpdater.h"
#include "PFCore/partflow/updaters/strategies/MagnitudeUpdater.h"
#include "PFGpu/partflow/GpuParticleUpdater.h"
#include "PFCore/stats/PerformanceTracker.h"
#include "PFCore/partflow/vectorFields/ParticleFieldVolume.h"

using namespace PFCore::math;
using namespace PFCore::input;
using namespace PFCore::partflow;
using namespace std;

void printParticleSet(const ParticleSetView& view, bool printVelocity = false, int step = 0);

int main(int argc, char** argv) {
	
	for (int i = 32; i < 1024; i += 32)
	{
		int numParticles = 1024 * i;

		cout << "Num Particles: " << numParticles << endl;
		cout << "--------------------------" << endl;

		for (int gpuId = -1; gpuId < 4; gpuId++)
		{
			GpuParticleFactory psetFactory;
			ParticleSetRef localSet = psetFactory.createLocalParticleSet(numParticles, 0, 0, 1);
			ParticleSetRef deviceSet = psetFactory.createParticleSet(gpuId, numParticles, 0, 0, 1);

			vec4 startField = vec4(0.0f, 0.0f);
			vec4 lenField = vec4(100.0f, 100.0f, 100.0f, 1.0f);
			vec4 sizeField = vec4(500, 500, 100, 1);
			ParticleFieldRef localField = psetFactory.createLocalParticleField(0, 1, startField, lenField, sizeField);
			ParticleFieldRef deviceField = psetFactory.createParticleField(gpuId, 0, 1, startField, lenField, sizeField);

			for (int f = 0; f < sizeField.x*sizeField.y*sizeField.z; f++)
			{
				localField->getVectors(0)[f] = 5.0f*(float(std::rand()) / RAND_MAX - 0.5f)*2.0f;
			}
			deviceField->copy(*localField);

			GpuEmitterFactory emitterFactory;
			EmitterRef emitter = EmitterRef(emitterFactory.createBoxEmitter(startField, lenField, 1));

			emitter->emitParticles(*deviceSet, 0, true);

			localSet->copy(*deviceSet);
			//printParticleSet(*localSet, false, 0);

			partFlowCounterGetCounter("advect")->reset();

			AdvectorRef advector = AdvectorRef(new GpuVectorFieldAdvector<RungaKutta4<ParticleFieldVolume>, ParticleFieldVolume>(RungaKutta4<ParticleFieldVolume>(), ParticleFieldVolume(*deviceField, 0)));
			float dt = 0.1f;
			for (int f = 0; f < (gpuId < 0 ? 1 : 10); f++)
			{
				partFlowCounterStart("advect");
				advector->advectParticles(*deviceSet, f, dt*float(f), dt);
				partFlowCounterStop("advect");
			}


			localSet->copy(*deviceSet);
			//printParticleSet(*localSet, false, 0);

			cout << gpuId << ": " << partFlowCounterGetCounter("advect")->getAverage() << endl;
		}

		cout << endl;
	}

	int test;
	cin >> test;

	return 0;
}

void printParticleSet(const ParticleSetView& view, bool printVelocity, int step)
{
	cout << "-------- Step ";
	cout << step << " ---------" << endl;

	for (int f = 0; f < 10; f++)//view.getNumParticles(); f++)
	{
		const vec3& pos = view.getPosition(f, step);
		cout << pos.x << "," << pos.y << "," << pos.z << endl;
	}

	if (printVelocity)
	{
		for (int f = 0; f < 10; f++)//view.getNumParticles(); f++)
		{
			const vec3& vel = view.getVector(0, f, step);
			cout << "Vel: " << vel.x << "," << vel.y << "," << vel.z << endl;
		}
	}
}