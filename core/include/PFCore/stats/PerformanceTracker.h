/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef PERFORMANCETRACKER_H_
#define PERFORMANCETRACKER_H_

#include <memory>
#include <map>
#include "PFCore/stats/PerformanceCounter.h"
#include <mutex>

#define partFlowCounterStart(counter) PFCore::stats::PerformanceTracker().instance()->start(counter)
#define partFlowCounterStop(counter) PFCore::stats::PerformanceTracker().instance()->stop(counter)
#define partFlowCounterGetCounter(counter) PFCore::stats::PerformanceTracker().instance()->getCounter(counter)

namespace PFCore {
namespace stats {

class PerformanceTracker;
typedef std::shared_ptr<PerformanceTracker> PerformanceTrackerRef;

class PerformanceTracker {
public:
	PerformanceTracker();
	virtual ~PerformanceTracker();

	void start(const std::string& counter);
	void stop(const std::string& counter);

	static PerformanceTrackerRef instance();

	PerformanceCounterRef getCounter(const std::string& counter);

private:
	std::map<std::string, PerformanceCounterRef> _counters;
	std::mutex _counterMutex;
};

} /* namespace stats */
} /* namespace PFCore */

#endif /* PERFORMANCETRACKER_H_ */
