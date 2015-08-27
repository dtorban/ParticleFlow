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
#include <iostream>



namespace PFCore {
namespace stats {

class PerformanceCounter;
typedef std::shared_ptr<PerformanceCounter> PerformanceCounterRef;

class PerformanceCounter {
public:
	PerformanceCounter(const std::string& key, const std::string& name);
	virtual ~PerformanceCounter();

	std::string getKey() const;
	std::string getName() const;
	double getTotal() const;
	double getAverage() const;
	double getNumValues() const;

	void add(double val);
	void start();
	void stop();
	void reset();

private:
	std::string _key;
	std::string _name;
	double _total;
	long _num;
	unsigned long _start;
	bool _started;
};

inline std::ostream& operator<<(std::ostream& os, const PerformanceCounter& counter)
{
    os << counter.getKey() << ", " << counter.getTotal() << ", " << counter.getNumValues() << ", " << counter.getAverage();
    return os;
}

} /* namespace stats */
} /* namespace PFCore */

#endif /* PERFORMANCECOUNTER_H_ */
