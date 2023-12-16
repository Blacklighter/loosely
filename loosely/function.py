import utils

def get_classes():
    import atom, process, tree, dataset
    modules = [atom, process, tree, dataset]
    classes = []

    def _filter(c):
        base_classes = [atom.LooseAtom, tree.LooseTree]
        return any([issubclass(c, base_class) and c != base_class for base_class in base_classes])

    for module in modules:
        classes.extend(utils.get_all_classes_in_module(module, _filter))
    return classes

CLASSES = get_classes()

def get_class_by_path(path):
    for c in CLASSES:
        if c.is_matched_from_path(path):
            return c
    else:
        raise NotImplementedError(f"There is no implementation for this kind of path: {path}")
    

if __name__ == '__main__':
    pass
    # print(get_class_by_path('D:/documents/AcademicDocuments/customed_python_pkgs/loosely/example_dataset3.nids/table/53/.meta.json'))