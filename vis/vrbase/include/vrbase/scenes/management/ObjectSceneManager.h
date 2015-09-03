/*
 * Copyright Regents of the University of Minnesota, 2015.  This software is released under the following license: http://opensource.org/licenses/GPL-2.0
 * Source code originally developed at the University of Minnesota Interactive Visualization Lab (http://ivlab.cs.umn.edu).
 *
 * Code author(s):
 * 		Dan Orban (dtorban)
 */

#ifndef OBJECTSCENEMANAGER_H_
#define OBJECTSCENEMANAGER_H_

#include <memory>
#include <vector>
#include <map>
#include "vrbase/scenes/management/ObjectSceneFactory.h"

namespace vrbase {

template<typename ObjectTypeRef, typename SceneType, typename FactoryType>
class ObjectSceneManager {
public:
	ObjectSceneManager(const std::vector<ObjectTypeRef>* objects, FactoryType factory);
	virtual ~ObjectSceneManager();

	void refresh();
	std::shared_ptr<SceneType> get(ObjectTypeRef object);

private:
	const std::vector<ObjectTypeRef>* _objects;
	std::map<ObjectTypeRef, std::shared_ptr<SceneType> > _sceneMap;
	FactoryType _factory;
};

template<typename ObjectTypeRef, typename SceneType, typename FactoryType>
inline ObjectSceneManager<ObjectTypeRef, SceneType, FactoryType>::ObjectSceneManager(
		const std::vector<ObjectTypeRef>* objects, FactoryType factory) : _objects(objects), _factory(factory) {
}

template<typename ObjectTypeRef, typename SceneType, typename FactoryType>
inline ObjectSceneManager<ObjectTypeRef, SceneType, FactoryType>::~ObjectSceneManager() {
}

template<typename ObjectTypeRef, typename SceneType, typename FactoryType>
inline void ObjectSceneManager<ObjectTypeRef, SceneType, FactoryType>::refresh() {

	std::map<ObjectTypeRef, std::shared_ptr<SceneType> > newMap;

	// Create new map
	for (int f = 0; f < _objects->size(); f++)
	{
		ObjectTypeRef object = (*_objects)[f];
		typename std::map<ObjectTypeRef, std::shared_ptr<SceneType>>::iterator it = _sceneMap.find(object);
		if (it != _sceneMap.end())
		{
			std::shared_ptr<SceneType> scene = it->second;
			newMap[object] = scene;
		}
		else
		{
			std::shared_ptr<SceneType> scene = _factory.create(object);
			newMap[object] = scene;
			scene->init();
		}
	}

	_sceneMap = newMap;
}

template<typename ObjectType, typename SceneType, typename FactoryType>
inline std::shared_ptr<SceneType> ObjectSceneManager<ObjectType, SceneType, FactoryType>::get(
		ObjectType object) {
	return _sceneMap[object];
}

template<typename ObjectTypeRef, typename SceneType>
class SceneManager : public ObjectSceneManager<ObjectTypeRef, SceneType, ObjectSceneFactory<ObjectTypeRef, SceneType>>
{
public:
	SceneManager(const std::vector<ObjectTypeRef>* objects) :
		ObjectSceneManager<ObjectTypeRef, SceneType, ObjectSceneFactory<ObjectTypeRef, SceneType>>
		(objects, ObjectSceneFactory<ObjectTypeRef, SceneType>()) {}
	virtual ~SceneManager() {}
};

} /* namespace vrbase */


#endif /* OBJECTSCENEMANAGER_H_ */
