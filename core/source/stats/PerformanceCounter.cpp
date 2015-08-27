/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#include <PFCore/stats/PerformanceCounter.h>

#include <chrono>
#include <iostream>

namespace PFCore {
namespace stats {

PerformanceCounter::PerformanceCounter(const std::string& key, const std::string& name) : _key(key), _name(name), _total(0.0f), _num(0), _start(0.0f), _started(false) {
}

PerformanceCounter::~PerformanceCounter() {
}

double PerformanceCounter::getTotal() {
	return _total;
}

double PerformanceCounter::getAverage() {
	return _num > 0 ? _total/_num : 0.0f;
}

void PerformanceCounter::add(double val) {
	_total += val;
	_num++;
}

void PerformanceCounter::start() {
	_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
	_started = true;
}

void PerformanceCounter::stop() {

	if (_started)
	{
		unsigned long end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		add(end-_start);
		_started = false;
	}
}

} /* namespace stats */
} /* namespace PFCore */
