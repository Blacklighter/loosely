import os, utils, datetime, json, re, shutil
from LooseAtom import LooseAtom
from abc import ABC, abstractstaticmethod, abstractmethod, abstractproperty
from glob import glob
import inspect

def get_class_by_name(name):
    CLASS_NAME_MAP = {
        'LooseDict': LooseDict
    }
    assert name in CLASS_NAME_MAP
    return CLASS_NAME_MAP[name]

class LooseIterable(ABC):
    @abstractproperty
    def iter_obj_type(self):
        pass

    @property
    def dir(self):
        return self._dir
    
    @abstractmethod
    def iter_items(self):
        pass

    def auto_update_items(self):
        for k, v in self.iter_items():
            if isinstance(v, LooseAtom):
                if self.dir is None:
                    v.save_path = None
                else:
                    v.save_path = os.path.join(self.dir, f'{k}.json')
            else:
                # Is LooseIterable type
                v.dir = os.path.join(self.dir, k)

    @dir.setter
    def dir(self, new_dir):
        self._dir = new_dir
        self.auto_update_items()

    @property
    def _default_meta(self):
        return {
            'type': self.__class__.__name__
        }

    @property
    def meta_file_exists(self):
        return self.dir is not None and os.path.isfile(self.get_meta_file_path(self.dir))

    @staticmethod
    def is_loose_dir(dir_path):
        return os.path.isdir(dir_path) and os.path.isfile(LooseIterable.get_meta_file_path(dir_path))

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, new_meta):
        self._meta = new_meta
        if self.meta_file_exists:
            self.save_meta()

    @staticmethod
    def get_meta_file_path(meta_dir):
        return os.path.join(meta_dir, '.meta.json')
    
    def save_meta(self):
        return utils.save_json(self.get_meta_file_path(self.dir), self._meta)
    
    def load_meta(self):
        meta = utils.load_json(self.get_meta_file_path(self.dir))
        self._meta = meta
        return meta

    def __init__(self, dir_path=None, meta=None):
        self.iter_obj = self.iter_obj_type()
        self.dir = dir_path
        self.meta = meta if meta is not None else self._default_meta

    @staticmethod
    def _check_value_type(v):
        assert isinstance(v, LooseIterable) or isinstance(v, LooseAtom)

    def clear(self):
        utils.remove_all_files_in_dir(self.dir)

    def _check_dir(self):
        if self.dir is None:
            raise Exception('Have not set the root dir yet')
        elif type(self.dir) != str:
            raise Exception(f'Wrong root dir type: {type(self.dir)}')
        elif type(self.dir) == str and not os.path.isdir(os.path.dirname(self.dir)):
            raise Exception(f"Parent direcotry doesn't exist: {os.path.dirname(self.dir)}")
    
    def save_items(self):
        for k, v in self.iter_items():
            v.save_to_disk()

    def save_to_disk(self, new_dir=None):
        if new_dir is not None:
            self.dir = new_dir
        self._check_dir()
        if os.path.isdir(self.dir):
            self.clear()
        else:
            os.mkdir(self.dir)  
        self.save_meta()
        self.save_items()        

    @classmethod
    def load_from_disk(cls, load_dir):
        assert os.path.isdir(load_dir)
        root = cls(load_dir)
        root.load_meta()
        assert get_class_by_name(root.meta['type']) == cls
        for path in glob(os.path.join(load_dir, '*')):
            name = os.path.basename(path)
            if cls.is_loose_dir(path):
                child_meta = utils.load_json(cls.get_meta_file_path(path))
                child_class = get_class_by_name(child_meta['type'])
                child = child_class.load_from_disk(path)
                root.iter_obj[name] = child
            elif utils.get_ext(path) == '.json':
                root.iter_obj[name] = LooseAtom(save_path=path, load_path=path)
            else:
                continue
        return root

    def __str__(self):
        result_dict = dict()
        result_dict['type'] = self.meta['type']
        result_dict['dir'] = self.dir
        result_dict['children'] = dict()
        for k, v in self.iter_items():
            if isinstance(v, LooseAtom):
                result_dict['children'][k] = {'type': v.__class__.__name__, 'state': v.state, 'data': v.data, 'load_path': v.load_path}
            else:
                result_dict['children'][k] = json.loads(str(v))
        return json.dumps(result_dict, indent=2)
    
    def __len__(self):
        return len(self.iter_obj)
    
    def __contains__(self, k):
        return k in self.iter_obj

    def __getitem__(self, index):
        return self.iter_obj[index]
    
    def __delitem__(self, index):
        if self.dir is not None:
            path = os.path.join(self.dir, index)
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.isfile(path):
                    os.remove(path)

        del self.iter_obj[index]

    def __setitem__(self, k, v):
        self._check_value_type(v)

        if k in self:
            del self[k]

        if self.dir is not None:
            path = os.path.join(self.dir, k)
            if isinstance(v, LooseAtom):
                v.save_path = path
            else:
                v.dir = path
                
            if os.path.isdir(self.dir):
                v.save_to_disk()

        self.iter_obj[k] = v



class LooseDict(LooseIterable):
    @property
    def iter_obj_type(self):
        return dict

    def iter_items(self):
        return self.iter_obj.items()

# test_root_dir = 'D:/documents/AcademicDocuments/customed_python_pkgs/loosely/test_loose_dict'
# root = LooseDict.load_from_disk(test_root_dir)
# root['child2']['test.json'] = LooseAtom(555)
# child2 = LooseDict()
# root['child2'] = child2
# print(root)

class LooseList(LooseIterable):
    @property
    def iter_obj_type(self):
        return dict

    def iter_items(self):
        return enumerate(self.iter_obj)
    
# import sys
# print(sys.modules[__file__])
    
class LooseArrayElement(LooseAtom):
    def auto_file_update(func):
        def getter(self):
            return getattr(self, '_'+func.__name__)
        def setter(self, value):
            try:
                old_path = self.file_path
                old_exists = old_path is not None and os.path.isfile(old_path)
            except:
                setattr(self, '_'+func.__name__, value)
                return
            
            if old_exists:
                os.remove(old_path)
            setattr(self, '_'+func.__name__, value)
            self.save_path = self.file_path
            if self._check() and old_exists:
                self.save_to_disk()
        return property(getter, setter)

    @auto_file_update
    def dir(self):
        pass

    @auto_file_update
    def index(self):
        pass

    def __init__(self, data, save_dir=None, index=None):
        self.dir = save_dir
        self.index = index
        super().__init__(data, self.file_path)

    def _check(self):
        if self.dir is None and self.index is None:
            return False
        elif self.dir is not None and self.index is not None and os.path.isdir(self.dir) and type(self.index) == int:
            return True
        else:
            raise Exception(f'LooseArrayElement checking failed: dir={self.dir}, index={self.index}')

    @staticmethod
    def get_file_name(index):
        return f'{index}.json'
    
    @property
    def file_name(self):
        assert self.index is None or type(self.index) == int
        return self.get_file_name(self.index) if self.index is not None else None

    @staticmethod
    def get_file_path(file_dir, index):
        return os.path.join(file_dir, LooseArrayElement.get_file_name(index))

    @property
    def file_path(self):
        return self.get_file_path(self.dir, self.index) if self._check() else None
    
    @classmethod
    def load_from_disk(cls, save_dir, index):
        arr = cls.__base__.load_from_disk(cls.get_file_path(save_dir, index))
        return cls(arr.data, save_dir, index)
        # obj = cls(save_dir, index)

        # print(cls.__base__)

# ele_dir = 'D:/documents/AcademicDocuments/customed_python_pkgs/loosely/test'
# ele_index = 6
# ele_data = {'test': 11111}
# ele = LooseArrayElement(ele_data, ele_dir, index=ele_index)
# ele.save_to_disk()
# ele = LooseArrayElement.load_from_disk(ele_dir, ele_index)
# print(ele, type(ele))
# ele.index = 6
# ele.dir = 'D:/documents/AcademicDocuments/customed_python_pkgs/loosely'
# print(ele, type(ele))


# class LooseArray(LooseNamedObj):
#     def __init__(self, dir_path=None, name=None):
#         super().__init__(dir_path, name)
#         self.elements = []

# arr_dir = 'D:/documents/AcademicDocuments/customed_python_pkgs/loosely/test_arr'
# arr = LooseArray(arr_dir)
# print(arr)
# l_arr = LooseArray.load_from_disk(obj_dir)

# print(l_arr)

# print(~(None^None))