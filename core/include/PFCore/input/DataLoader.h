/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef DATALOADER_H_
#define DATALOADER_H_

#include <memory>

namespace PFCore {
namespace input {

class DataLoader;
typedef std::shared_ptr<class DataLoader> DataLoaderRef;

class DataLoader {
public:
	virtual ~DataLoader() {}

	virtual bool load(float *data, int size) = 0;
};

}
}

#endif /* DATALOADER_H_ */
