/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef OBJECTSCENEFACTORY_H_
#define OBJECTSCENEFACTORY_H_

#include <memory>

namespace vrbase {

template<typename ObjectTypeRef, typename SceneType>
class ObjectSceneFactory {
public:
	ObjectSceneFactory() {}
	virtual ~ObjectSceneFactory() {}

	std::shared_ptr<SceneType> create(ObjectTypeRef object)
	{
		return std::shared_ptr<SceneType>(new SceneType(object));
	}
};

} /* namespace vrbase */


#endif /* OBJECTSCENEFACTORY_H_ */
