/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef PERFORMANCECOUNTER_H_
#define PERFORMANCECOUNTER_H_

#include <string>
#include <memory>

namespace PFCore {
namespace stats {

class PerformanceCounter;
typedef std::shared_ptr<PerformanceCounter> PerformanceCounterRef;

class PerformanceCounter {
public:
	PerformanceCounter(const std::string& key, const std::string& name);
	virtual ~PerformanceCounter();

	double getTotal();
	double getAverage();
	double getNumValues();

	void add(double val);
	void start();
	void stop();

private:
	std::string _key;
	std::string _name;
	double _total;
	long _num;
	unsigned long _start;
	bool _started;
};

} /* namespace stats */
} /* namespace PFCore */

#endif /* PERFORMANCECOUNTER_H_ */
