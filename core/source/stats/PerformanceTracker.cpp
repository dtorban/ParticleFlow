/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <PFCore/stats/PerformanceTracker.h>

#include <iostream>

namespace PFCore {
namespace stats {

PerformanceTracker::PerformanceTracker() {
}

PerformanceTracker::~PerformanceTracker() {
}

PerformanceTrackerRef PerformanceTracker::instance() {
	static PerformanceTrackerRef instance(new PerformanceTracker());
	return instance;
}

void PerformanceTracker::start(const std::string& counter) {
	getCounter(counter)->start();
}

void PerformanceTracker::stop(const std::string& counter) {
	getCounter(counter)->stop();
}

PerformanceCounterRef PerformanceTracker::getCounter(
		const std::string& counter) {

	_counterMutex.lock();
	std::map<std::string,PerformanceCounterRef>::iterator it = _counters.find(counter);

	PerformanceCounterRef perfCounter;
	if(it != _counters.end())
	{
		perfCounter = it->second;
	}
	else
	{
		perfCounter = PerformanceCounterRef(new PerformanceCounter(counter, counter));
		_counters[counter] = perfCounter;

	}

	_counterMutex.unlock();

	return perfCounter;
}

} /* namespace stats */
} /* namespace PFCore */
