def get_object(objects, name: object = None) -> object:
    '''Factory'''
    return objects[name]()

