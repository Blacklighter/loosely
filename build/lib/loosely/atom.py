# import numpy as np, torch
# from numpy import random
# import os, json, re, sys
# from uuid import uuid1
# from collections.abc import Iterable

import json, os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, '.')))
from loosely.base import LooseBase, FileType
from loosely.utils import get_keys
import numpy as np
import re
# print(os.path.abspath(os.path.join(__file__, '..', '..')))
# from . import utils
# from zipfile import ZipFile
class LooseAtom(LooseBase):
    pass


class LooseAtomDict(LooseAtom):
    EXT = '.json'
    SUPPORT_INPUT_DATA_TYPE = [dict, list, tuple]
    CLS_ALIAS = 'atom_dict'


    def __init__(self, data=dict(), read_only=False):
        super().__init__(data, read_only)

    @classmethod
    def _load_data(cls, path, read_only=False):
        with open(path) as f:
            return json.load(f)

    def _save_data(self, path):
        with open(path, mode='w') as f:
            json.dump(self.data, f, indent=2)

    @property
    def extra_str(self):
        key_num = len(list(get_keys(self.get_data())))
        return super().extra_str + f"{key_num} key{'s' if key_num >= 2 else ''}"
    
    @classmethod
    def extract_extra_from_str(cls, extra_str):
        inner_text = extra_str.replace(' key', '').replace('s', '')
        return {
            'key_num': int(inner_text)
        }

    @classmethod
    def _check_key(cls, key):
        assert type(key) == str

    def __getitem__(self, key):
        self._check_key(key)
        return self.get_data()[key]
    
    def __setitem__(self, key, item):
        self.check_read_only()
        self._check_key(key)
        if self.data is None and self.can_load:
            data = self.get_data()
            data[key] = item
            self.data = data
            self.save(verbose=False)
        elif self.data is not None:
            self.data[key] = item

class LooseAtomArray(LooseAtom):
    EXT = '.npy'
    SUPPORT_INPUT_DATA_TYPE = [list, tuple, np.ndarray]
    CLS_ALIAS = 'atom_arr'

    def __init__(self, data=list(), read_only=False):
        super().__init__(np.array(data), read_only)

    @classmethod
    def _load_data(cls, path, read_only=False):
        return np.load(path, allow_pickle=True)

    def _save_data(self, path):
        np.save(path, self.data)

    @property
    def extra_str(self):
        shape = self.get_data().shape
        return super().extra_str + f"shape:({'*'.join([str(i) for i in shape])})"
    
    @classmethod
    def extract_extra_from_str(cls, extra_str):
        groups = re.match(r'shape:\((.*)\)', extra_str).groups()
        inner_text = groups[0] if len(groups) > 0 else ''
        return {
            'shape': [int(i) for i in inner_text.split('*') if i != '']
        }
    
    def to_dict(self):
        r = super().to_dict()
        r['shape'] = tuple(self.get_data().shape)
        return r

import pickle

class LooseAtomObj(LooseAtom):
    EXT = ''
    CLS_ALIAS = 'atom_obj'

    def __init__(self, data, read_only=False):
        super().__init__(data, read_only)

    @classmethod
    def _load_data(cls, path, read_only=False):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _save_data(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)

def _func():
    pass
class LooseAtomFunc(LooseAtomObj):
    EXT = '.func'
    SUPPORT_INPUT_DATA_TYPE = _func.__class__
    CLS_ALIAS = 'atom_func'

import nibabel as nib
from pathlib import Path

class LooseAtomNifti(LooseAtom):
    EXT = '.nii'
    SUPPORT_INPUT_DATA_TYPE = nib.Nifti1Image
    CLS_ALIAS = 'atom_nii'

    def __init__(self, data=list(), affine=None, read_only=False):
        if type(data) == str:
            if not Path(data).is_file():
                raise FileNotFoundError(f"No such a nifti file path for '{self.__class__.__name__}': {data}")
            img = nib.load(data)
        elif isinstance(data, nib.Nifti1Image):
            img = data
        else:
            if affine is None:
                raise ValueError(f"Affine should be given when initiate '{self.__class__.__name__}' with array data.")
            img = nib.Nifti1Image(np.array(data), np.array(affine))
        super().__init__(img, read_only)

    @classmethod
    def _check_path_basically(cls, path):
        exts1 = Path(cls.get_file_name_from_stem('tmp')).suffixes
        exts2 = Path(cls.get_file_name_from_stem('tmp')).suffixes+['.gz']

        def _func(exts, path_exts):
            return not (len(path_exts) >= len(exts) and path_exts[-len(exts):] == exts)

        path_exts = Path(path).suffixes
        if _func(exts1, path_exts) and _func(exts2, path(exts2)):
            raise ValueError(f"Suffixes of target path '{''.join(path_exts)}' is not matched for '{''.join(exts1)}' and '{''.join(exts2)}': {path}")

    @classmethod
    def _load_data(cls, path, read_only=False):
        return nib.load(path)

    def _save_data(self, path):
        self.data.to_filename(path)

    @property
    def extra_str(self):
        shape = self.get_data().get_fdata().shape
        return super().extra_str + f"shape:({'*'.join([str(i) for i in shape])})"
    
    @classmethod
    def extract_extra_from_str(cls, extra_str):
        return LooseAtomArray.extract_extra_from_str(extra_str)
    
    def to_dict(self):
        r = super().to_dict()
        r['shape'] = tuple(self.get_data().get_fdata().shape)
        return r



# def generate_ext_match_pattern_by_simple_list(l):
#     return '^('+'|'.join([fr'{item}'.replace('.', r'\.') for item in l])+')$'

# class LooseAtom(LooseBase):
#     @abstractclassmethod
#     def get_support_ext(cls):
#         pass

#     @classmethod
#     def get_ext_match_pattern(cls):
#         try:
#             ext = cls.get_support_ext()
#         except:
#             ext = cls.get_support_ext(cls)
#         finally:
#             return generate_ext_match_pattern_by_simple_list([ext])

#     @classmethod
#     def is_matched_from_path(cls, path):
#         return re.match(cls.get_ext_match_pattern(), utils.get_ext(path)) is not None
    
#     @property
#     def state_str(self):
#         result_str = ''
#         can_save, can_load = self.can_save, self.can_load
#         result_str += 'Not in memory, ' if not self.data_loaded else 'In memory, '
#         save_str = ('can' if can_save else 'cannot') + ' save'
#         load_str = ('can' if can_load else 'cannot') + ' load'

#         divider = ' but ' if can_save^can_save else ' and '
#         result_str += save_str + divider + load_str
#         return result_str

#     def __init__(self, data=None, save_path=None, load_path=None):
#         # keep in memory
#         self.data = data
#         self.save_path = save_path
#         self.load_path = load_path
#         self.data_loaded = False
    
#     @classmethod
#     def _check_path_to_save(cls, path):
#         print(path)
#         if path is None:
#             raise Exception('Path have not been set.')
#         elif type(path) != str:
#             raise Exception(f'Save path is a not string: {type(path)}')
#         elif not os.path.isdir(os.path.dirname(path)):
#             raise Exception(f"The parent dierectory doesn't exist: {os.path.dirname(path)}")
#         elif not cls.is_matched_from_path(path):
#             raise Exception(f"Wrong extension of save path for '{cls.__name__}': {utils.get_ext(path)}")

#     @staticmethod
#     def _throw_exception_for_data_check(data):
#         raise Exception(f'Data checking failed: {type(data)}')

#     @classmethod
#     def _check_data_roughly(cls, data):
#         return
    
#     @staticmethod
#     def _wrap_data(data, extra_obj={}):
#         return data

#     @abstractclassmethod
#     def _save_data(self, path, wrarpped_data, extra_obj):
#         pass

#     def get_path_to_save(self):
#         try:
#             self._check_path_to_save(self.save_path)
#             return self.save_path
#         except:
#             self._check_path_to_save(self.load_path)
#             return self.load_path
        
#     @property
#     def can_save(self):
#         try:
#             self.get_path_to_save()
#             return True
#         except:
#             return False
        
#     @property
#     def can_load(self):
#         try:
#             self._check_path_to_save(self.load_path)
#             return os.path.isfile(self.load_path)
#         except:
#             return False

#     def save_to_disk(self, save_path=None):
#         path = save_path if save_path is not None else self.get_path_to_save()
#         data = self.load_data(path) if self.data is None and self.can_load else self.data
#         self.release_data()
#         self._check_data_roughly(data)
#         extra_obj = {}
#         self._save_data(path, self._wrap_data(data, extra_obj), extra_obj)
#         self.path = path
#         self.load_path = path

#     def release_data(self):
#         del self.data
#         self.data = None
#         self.data_loaded = False

#     def __str__(self):
#         return json.dumps({
#             'state_str': self.state_str,
#             'can_save': self.can_save,
#             'can_load': self.can_load,
#             'data': str(self.data)
#         }, indent=2)
    
#     @abstractclassmethod
#     def _load_data(cls, path):
#         pass
    
#     @staticmethod
#     def _unwrap_data(data, extra_param):
#         return data

#     def _get_extra_param_in_loading(self, path):
#         return None

#     def load_data(self, path):
#         return self._unwrap_data(self._load_data(path), self._get_extra_param_in_loading(path))

#     @classmethod
#     def load_from_disk(cls, load_path=None):
#         obj = cls(save_path=load_path, load_path=load_path)
#         obj.reload_data()
#         return obj
    
#     def reload_data(self):
#         self.data = self.get_data(force_load=True)
#         self.data_loaded = True
#         return self.data
    
#     def get_data(self, force_load=False):
#         if force_load or (not force_load and not self.data_loaded):
#             assert self.can_load
#             return self.load_data(self.load_path)
#         else:
#             return self.data
    
#     def __getitem__(self, index):
#         data = self.data if self.data is not None else self.load_data(self.load_path)
#         return data[index]
    
#     def __setitem__(self, index, new_v):
#         data = self.data if self.data is not None else self.load_data(self.load_path)
#         data[index] = new_v
#         if self.load_path is not None and (self.save_path is None or self.save_path == self.load_path):
#             self.data = data
#             self.save_to_disk()

#     def update_data(self, obj, auto_save=True):
#         if isinstance(obj, LooseAtom):
#             self.data = obj.data

#         if auto_save:
#             self.save_to_disk()


# class LooseJsonAtom(LooseAtom):
#     @classmethod
#     def get_support_ext(cls):
#         return '.json'
    
#     @classmethod
#     def _save_data(self, path, wrarpped_data, extra_obj):
#         utils.save_json(path, wrarpped_data)

#     def _load_data(cls, path):
#         return utils.load_json(path)
    
#     def update_data(self, obj, partial=True, auto_save=True):
#         if not self.data_loaded and self.can_load:
#             self.reload_data()
#             self.update_data(obj, partial, auto_save)
#             self.release_data()
#         else:
#             if type(obj) == dict and partial:
#                 self.data.update(obj)
#             else:
#                 self.data = obj
#             super().update_data(obj, auto_save)
    
# # data = {
# #     'a': 'fdadfa',
# #     'b': {
# #         'tes': [1,3,2],
# #         'c': (1,2,3),
# #     },
# #     'd': False,
# #     'e': None
# # }
# # # save_path = 'D:/documents/AcademicDocuments/customed_python_pkgs/loosely/test/test.json'
# # atom = LooseJsonAtom(load_path=save_path)
# # # print(atom.data_loaded)
# # # print(atom)
# # atom.update_data({'a': 123})
# # print(atom)
# # print(atom)
# # atom.reload_data()
# # print(atom)
# # print(atom.data)
# # atom.save_to_disk()

# class LooseBundleAtom(LooseAtom):
#     @classmethod
#     def get_support_ext(cls):
#         return '.zip'

#     @staticmethod
#     def _wrap_data(data, extra_obj={}):
#         if type(data) == dict:
#             for k, v in data.items():
#                 data[k] = LooseBundleAtom._wrap_data(v, extra_obj)
#         elif type(data) in [list, tuple, np.ndarray, torch.Tensor]:
#             arr = np.asarray(data)
#             name = uuid1()
#             file_name = f'{name}.npy'
#             assert np.issubdtype(arr.dtype, np.number)
#             extra_obj[file_name] = arr
#             return f"<numpy.ndarray dtype='{arr.dtype}' id='{name}'>"
#         elif type(data) == set:
#             return LooseBundleAtom._wrap_data(list(data), extra_obj)
#         return data

#     @classmethod
#     def _save_data(self, path, wrapped_data, extra_obj):
#         with ZipFile(path, mode='w') as zip_file:
#             for file_name, file_data in extra_obj.items():
#                 with zip_file.open(file_name, 'w') as extra_file:
#                     ext = utils.get_ext(file_name)
#                     if ext == '.npy':
#                         np.save(extra_file, file_data)
#                     else:
#                         raise Exception(f'Unsupported extension of zip file: {ext}')
#             with zip_file.open('data.json', 'w') as json_file:
#                 json_file.write(json.dumps(wrapped_data, indent=2).encode("utf-8"))


#     def _load_data(cls, path):
#         with ZipFile(path) as zip_file:
#             return json.loads(zip_file.read('data.json'))
    
#     OBJECT_PATTERN = re.compile(r"<(.+?)\s.*id='(.+)'>")

#     @staticmethod
#     def _unwrap_str(s, zip_file):
#         m = LooseBundleAtom.OBJECT_PATTERN.match(s)
#         if m:
#             obj_type_str, obj_id = m.groups()
#             if obj_type_str == 'numpy.ndarray':
#                 file_name = f'{obj_id}.npy'
#                 with zip_file.open(file_name) as np_file:
#                     return np.load(np_file)
#             else:
#                 raise Exception(f'Cannot unwrap the complex object type: {obj_type_str}')
#         else:
#             return s

#     def _get_extra_param_in_loading(self, path):
#         return ZipFile(path)

#     @staticmethod
#     def _unwrap_data(data, zip_file):
#         if isinstance(data, Iterable):
#             if type(data) == str:
#                 return LooseBundleAtom._unwrap_str(data, zip_file)
            
#             if type(data) == dict:
#                 iters = data.items()
#             else:
#                 iters = enumerate(data)
#             for k, v in iters:
#                 data[k] = LooseBundleAtom._unwrap_data(v, zip_file)
#             return data
#         else:
#             return data

# # data = {
# #     'a': 'fdadfa',
# #     'b': {
# #         'tes': [1,3,2],
# #         'c': (1,2,3),
# #     },
# #     'd': False,
# #     'e': None
# # }
# # save_path = 'D:/documents/AcademicDocuments/customed_python_pkgs/loosely/test/test.zip'

# # test_str = "<numpy.ndarray dtype='int32' id='1fd04735-698e-11ee-a76a-e02be9f58ff1'>"
# # print(LooseBundleAtom._unwrap_str(test_str, ZipFile(save_path)))

# # atom = LooseBundleAtom(load_path=save_path)
# # atom.save_to_disk()
# # d4d0ebb2-69a3-11ee-a48e-e02be9f58ff1
# # print(atom.reload_data())
# # atom['d'] = [0]
# # print(atom.data)
# # d = {
# #     'a': np.random.rand(121, 145, 121),
# #     'b': [random.randint(-10,10), random.randint(-10,10), random.randint(-10,10)],
# #     'c': str(uuid1()),
# #     'd': random.random()*100,
# #     'ff': False if random.random() > 0.5 else True,
# #     'dfa': None
# # }
# # print(utils.deep_find(d, lambda item: type(item) in [list, tuple, np.ndarray]))
# # atom = LooseBundleAtom(d, save_path=save_path)
# # atom.save_to_disk()
# # atom = LooseBundleAtom(load_path=save_path)
# # print(atom.reload_data())
# # print(atom)

# # s = "<numpy\.ndarray dtype='(.+)' id='(.+)'>"
# # def a(group):
# #     print(group)
# #     print(12313)
# #     return 'afda'
# # # print(re.sub(rf'{s}', a, str(s)))
# # ori = eval(f'r"{s}"')
# # print(s)
# # print()
# # print()

#         # if 
#         # result_data = data
# # x = {
# #     'fa': 1,
# #     'fff': None
# # }
# # save_path = 'D:/documents/AcademicDocuments/customed_python_pkgs/loosely/test.json'
# # # print(LooseAtom._check_obj(set([1,2,3])))
# # atom = LooseAtom(x, save_path)
# # atom.save_to_disk()
# # atom = LooseAtom.load_from_disk(save_path)
# # print(atom)
# # np.savetxt(save_path, np.array([9, '123', None]))
# # a = {'fda': 213}
# # for x in a:
# #     print(x)
# # path = 'D:/documents/AcademicDocuments/customed_python_pkgs/loosely/test/a.json'
# # print(get_class_by_path(path))
# # print(type(None) in [])
# # atom = LooseBundleAtom({}, save_path=path)
# # atom._check_path_to_save()

# # a = np.asarray([1, 5.0])
# # import torch
# # a = torch.Tensor([1, 5.0])
# # print(np.issubdtype(a.dtype, np.number))
# # print(a.dtype)

# # data = {
# #     'a': np.random.rand(121, 145, 121),
# #     'b': [random.randint(-10,10), random.randint(-10,10), random.randint(-10,10)],
# #     'c': str(uuid1()),
# #     'd': random.random()*100,
# #     'ff': False if random.random() > 0.5 else True,
# #     'dfa': None
# # }
# # save_path = 'D:/documents/AcademicDocuments/customed_python_pkgs/loosely/test/test.json'
# # extra_obj = {}
# # wrapped_data = LooseBundleAtom._wrap_data(data, extra_obj)
# # print(wrapped_data)
# # print(extra_obj)
# # LooseBundleAtom._save_data(save_path, wrapped_data)

# # s = "<numpy.ndarray dtype='float64' id='13a47863-696a-11ee-bcf6-e02be9f58ff1'>"
# # for group in LooseBundleAtom.NDARRAY_STR_PATTERN.match(s).groups():
# #     print(group)
# #     print(np.dtype(group) == np.float64)
# #     break