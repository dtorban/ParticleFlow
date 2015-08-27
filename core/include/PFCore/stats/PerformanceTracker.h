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
#include <iostream>

#define partFlowCounterStart(counter) PFCore::stats::PerformanceTracker().instance()->start(counter)
#define partFlowCounterStop(counter) PFCore::stats::PerformanceTracker().instance()->stop(counter)
#define partFlowCounterGetCounter(counter) PFCore::stats::PerformanceTracker().instance()->getCounter(counter)
#define partFlowCounterAdd(counter,amount) PFCore::stats::PerformanceTracker().instance()->getCounter(counter)->add(amount)

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
	friend std::ostream& operator<<(std::ostream& os, const PerformanceTracker& performanceTracker);

private:
	std::map<std::string, PerformanceCounterRef> _counters;
	std::mutex _counterMutex;
};

inline std::ostream& operator<<(std::ostream& os, const PerformanceTracker& performanceTracker)
{
	for(std::map<std::string,PerformanceCounterRef>::const_iterator it = performanceTracker._counters.begin(); it != performanceTracker._counters.end(); it++) {
	    os << *(it->second) << std::endl;
	}

    return os;
}

} /* namespace stats */
} /* namespace PFCore */

#endif /* PERFORMANCETRACKER_H_ */
