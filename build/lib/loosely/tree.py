import os, datetime, json, re, shutil
from abc import abstractstaticmethod, abstractmethod, abstractproperty
from loosely.base import LooseBase, FileType
from glob import glob
from pathlib import Path
import inspect, functools
import numpy as np
from loosely.utils import deep_map, simple_deep_copy, deep_copy, deep_iter_items, is_parent_trace, \
get_loosely_name_by_trace, extract_loosely_str, is_loosely_str, get_cls_from_alias, get_deep_item, set_deep_item, \
remove_by_path, get_keys, get_values, get_items, TRACE_DIVIDER
from loosely.atom import LooseAtomArray, LooseAtomDict, LooseAtom, LooseAtomNifti, LooseAtomFunc
from loguru import logger
from tqdm import tqdm
import nibabel as nib

class LooseTree(LooseBase):
    EXT = ''
    CLS_ALIAS = 'tree'

    @staticmethod
    def get_child_path(root_dir, child_name):
        return os.path.join(root_dir, child_name)

    @classmethod
    def check_path_to_load(cls, path):
        super().check_path_to_load(path)
        if not Path(path).is_dir():
            raise TypeError(f"{path} is not a directory for {cls.__name__}")
        
    def _before_save(self, save_path, allow_overwrite=True, verbose=True):
        save_path = super()._before_save(save_path, allow_overwrite, verbose)
        p = Path(save_path)
        if p.exists():
            if p.is_file():
                os.remove(p)
        else:
            os.mkdir(save_path)
        return save_path
    
    def get_children_paths(self):
        paths = []
        if self.can_load:
            for obj, trace, parent in deep_iter_items(self.get_data(), lambda x: isinstance(x, LooseBase) or is_loosely_str(x), iter_get_data=False):
                load_path = obj.load_path if isinstance(obj, LooseBase) else self.get_child_path(self.load_path, extract_loosely_str(obj)['file_name'])
                paths.append(load_path)
        return paths

    def _clean_excessive_children(self):
        if self.can_load:
            old_children_paths = [str(path.resolve()) for path in Path(os.path.join(self.load_path)).glob('*')] 
            children_paths = self.get_children_paths()
            # print(children_paths)
            excessive_paths = [old_path for old_path in old_children_paths if not (any([Path(new_path).samefile(old_path) for new_path in children_paths]))]
            for path in excessive_paths:
                remove_by_path(path)
    
    def _after_save(self, save_path, auto_release=True, verbose=True):
        super()._after_save(save_path, auto_release, verbose)
        # self._clean_excessive_children()

class LooseJSONTree(LooseTree):
    INFO_FILE_NAME = '.info.json'

    @classmethod
    def get_info_path(cls, dir_path):
        return os.path.join(dir_path, cls.INFO_FILE_NAME)

    @property
    def info_path(self):
        if not self.can_load:
            raise ValueError(f"load_path for '{self.__class__.__name__}' is invalid.")
        return self.get_info_path(self.load_path)
    
    @classmethod
    def check_path_to_load(cls, path):
        super().check_path_to_load(path)
        info_path = cls.get_info_path(path)
        try:
            LooseAtomDict.check_path_to_load(info_path)
        except FileNotFoundError:
            raise FileExistsError(f"Load failed: Cannot find the '{info_path}' file for '{cls.__name__}'")

    def get_children_paths(self):
        paths = []
        if self.can_load:
            paths = super().get_children_paths()
            paths.append(self.info_path)
        return paths

    @classmethod
    def _load_data(cls, path, read_only=False):
        info_path = cls.get_info_path(path)
        info_data = LooseAtomDict._load_data(info_path, read_only)
        for v, trace, parent in deep_iter_items(info_data, lambda x: is_loosely_str(x),  iter_get_data=False):
            info = extract_loosely_str(v)
            try:
                obj = info['type'].load_from_disk(cls.get_child_path(path, info['file_name']), read_only)
                obj.release()
                # print(23)
                # print('obj:', obj)
                # print('obj_data:', obj.get_data())
                # print(info)
                # print(parent)
                # print(123)
                parent[trace[-1]] = obj
            except FileNotFoundError:
                raise FileNotFoundError(f"Cannot find loosely item file for '{info['file_name']}'")
        # print('final')
        # print(type(info_data), info_data, 'meta' in info_data)
        return info_data

    @classmethod
    def _deep_save_loosely_obj(cls, dir_path, items, getter_func=None, parent_trace=[]):
        for obj, trace, parent in items:
            trace = parent_trace+trace
            for key in trace:
                cls._check_key_basically(key)
            if getter_func is not None:
                obj = getter_func(obj, trace, parent)
            # print(trace)
            file_name = obj.get_file_name_from_stem(get_loosely_name_by_trace(trace))
            save_path = os.path.join(dir_path, file_name)
            parent[trace[-1]] = obj.save(save_path, verbose=False)

    def _save_info(self, data=None, save_dir=None):
        data = self.get_data() if data is None else data
        save_dir = self.load_path if save_dir is None else save_dir
        for loosely_obj, trace, parent in deep_iter_items(data, lambda v: isinstance(v, LooseBase), iter_get_data=False):
            parent[trace[-1]] = loosely_obj.get_loosely_str()
        LooseAtomDict(data).save(self.get_info_path(save_dir), verbose=False)


    def _save_data(self, path, partial_save_keys=None, info_autosave=True):
        super()._save_data(path)
        # 首先保存所有的LooseAtom对象
        data = self.get_data()
        partial_save_keys = get_keys(data) if partial_save_keys is None else partial_save_keys
        # print(partial_save_keys)
        for key in partial_save_keys:
            self._check_key(key)
            item = data[key]
            if isinstance(item, LooseBase):
               data[key] = item.save(os.path.join(path, item.get_file_name_from_stem(key)), verbose=False)
            elif isinstance(item, np.ndarray):
                obj = LooseAtomArray(item)
                data[key] = obj.save(os.path.join(path, obj.get_file_name_from_stem(key)), verbose=False)
            else:
                self._deep_save_loosely_obj(path, deep_iter_items(item, lambda v: isinstance(v, LooseBase), 
                                                                  iter_get_data=False), parent_trace=[key])
                # self._deep_save_loosely_obj(path, deep_iter_items(item, lambda v: isinstance(v, LooseAtom), iter_get_data=False))
                self._deep_save_loosely_obj(path, deep_iter_items(item, lambda v: isinstance(v, np.ndarray), iter_get_data=False),                                                                
                    lambda arr, trace, parent: LooseAtomArray(arr), parent_trace=[key])
        for loosely_obj, trace, parent in deep_iter_items(data, lambda v: isinstance(v, LooseBase), iter_get_data=False):
            parent[trace[-1]] = loosely_obj.get_loosely_str()
        # print(data)
        if info_autosave:
            self._save_info(data=data, save_dir=path)

    def save(self, save_path=None, auto_release=True, allow_overwrite=True, partial_save_keys=None, info_autosave=True, verbose=True):
        save_path = self._before_save(save_path, allow_overwrite, verbose)
        self._save_data(save_path, partial_save_keys=partial_save_keys, info_autosave=info_autosave)
        self._after_save(save_path, auto_release=auto_release, verbose=verbose)
        return self.get_loosely_str(Path(save_path).stem)
    
    @classmethod
    def _check_key_basically(cls, key):
        if type(key) == str and TRACE_DIVIDER in key:
            raise IndexError(f"key '{key}' includes the trace divider: '{TRACE_DIVIDER}'")
        
    @classmethod
    def _check_key(cls, key):
        cls._check_key_basically(key)

    def __getitem__(self, key):
        self._check_key(key)
        return self.get_data()[key]
        # if is_loosely_str(v) and self.load_path is not None:
        #     info = extract_loosely_str(v)
        #     v = info['type'].load_from_disk(self.get_child_path(self.load_path, info['file_name']))
    
    @classmethod
    def _deep_del_loosely_obj_from_disk(cls, root):
        if isinstance(root, LooseBase) and root.can_load:
            root.remove_from_disk()
        else:
            for obj, trace, parent in deep_iter_items(root, 
                                        lambda item: isinstance(item, LooseBase) and item.can_load, iter_get_data=False):
                obj.remove_from_disk()
    def __setitem__(self, key, item):
        self.check_read_only()
        self._check_key(key)
        self.data = self.get_data()
        if key in self.data:
            old_item = self.data[key]
            self._deep_del_loosely_obj_from_disk(old_item)
        t = type(item)
        if t == np.ndarray:
            item = LooseAtomArray(item)
        if type(item) == LooseAtomFunc.SUPPORT_INPUT_DATA_TYPE:
            item = LooseAtomFunc(item)
        elif isinstance(item, nib.Nifti1Image):
            item = LooseAtomNifti(item)

        self.data[key] = item
        # print(self.data)
        if self.can_load:
            self.save(partial_save_keys=[key], verbose=False)
            # logger.info(f"New item for key '{key}' updated at {self.load_path}.")

    def __delitem__(self, key):
        self.check_read_only()
        self._check_key(key)
        del_item = self.get_data()[key]
        self.data = self.get_data()
        del self.data[key]
        self._deep_del_loosely_obj_from_disk(del_item)
        logger.info(f"The Key '{key}' for {self.__class__.__name__} deleted.")
        if self.can_load:
            self._save_info(data=self.data)
            # logger.info(f"Updated at {self.load_path}.")
            self.release()

    @classmethod
    def _create_iter_step(cls, obj, k, v, info_autosave=True):
        obj[k] = v

    @classmethod
    def _after_iter(cls, obj, dir_path, verbose):
        obj._save_info()
        obj.release()
        if verbose:
            cls._log_create_success(dir_path)

    @classmethod
    def create_from_iter(cls, dir_path, iterator, length=None, verbose=True, **init_kwargs):
        obj = cls(**init_kwargs)
        obj.save(dir_path, auto_release=False, info_autosave=True, verbose=False)
        obj.autoload()
        length_str = f" ({length} item{'s' if length > 1 else ''})" if length is not None else ''
        logger.info(f"{cls.__name__} initiate completely. Ready to create from iterator{length_str}.")
        loop = tqdm(iterator, desc=f'{cls.__name__} Creating', total=length)
        for k, v in loop:
            cls._create_iter_step(obj, k, v, info_autosave=False)
        cls._after_iter(obj, dir_path, verbose)
        return obj
    
    @classmethod
    def _log_create_success(cls, save_path):
        logger.success(f'{cls.__name__} create completely: {save_path}')


class LooseDict(LooseJSONTree):
    EXT = '.dict'
    SUPPORT_INPUT_DATA_TYPE = LooseAtomDict.SUPPORT_INPUT_DATA_TYPE
    CLS_ALIAS = 'tree_dict'

    def __init__(self, data={}, read_only=False):
        super().__init__(data, read_only)

    @property
    def extra_str(self):
        return LooseAtomDict(self.get_data()).extra_str

    @classmethod
    def extract_extra_from_str(cls, extra_str):
        return LooseAtomDict.extract_extra_from_str(extra_str)

    @classmethod
    def _check_key(cls, key):
        super()._check_key(key)
        LooseAtomDict._check_key(key)


class LooseArray(LooseJSONTree):
    EXT = '.arr'
    SUPPORT_INPUT_DATA_TYPE = [list, tuple]
    CLS_ALIAS = 'tree_arr'

    def __init__(self, data=list(), read_only=False):
        data = list(data)
        super().__init__(data, read_only)

    def __len__(self):
        return len(self.get_data())

    @property
    def extra_str(self):
        num = len(self.get_data())
        return super().extra_str + f"{num} item{'s' if num >= 2 else ''}"
    
    @classmethod
    def extract_extra_from_str(cls, extra_str):
        inner_text = extra_str.replace(' item', '').replace('s', '')
        return {
            'num': int(inner_text)
        }

    @classmethod
    def _check_key(cls, index, length=None):
        super()._check_key(index)
        if type(index) != int:
            raise TypeError(f"Wrong index type for '{cls.__name__}': {type(index)}")
        if length is not None:
            assert type(length) == int and length >= 0
            return index if index >= 0 else length+index
        

    def to_dict(self):
        r = super().to_dict()
        r['num'] = len(self)
        return r

    def __delitem__(self, index):
        index = self._check_key(index, len(self))
        super().__delitem__(index)
        if self.can_load:
            changed_indexes = list(range(index, len(self)))
            self.save(partial_save_keys=changed_indexes, verbose=False)
            logger.info(f'Left {len(changed_indexes)} items updated.')

    def insert(self, index, item, multiple=False, auto_release=True, info_autosave=True, verbose=True):
        self._check_key(index)
        old_length = len(self)
        if index > old_length:
            raise IndexError(f"The insert index '{index}' is too large (num: {old_length}).")
        data = self.get_data()
        heads = data[:index]
        tails = data[index:]
        if multiple:
            assert type(item) in [list, tuple]
            items = item
        else:
            items = [item]
        self.data = heads+items+tails
        if index <= old_length and self.can_load:
            changed_indexes = list(range(index, len(self.data)))
            self.save(partial_save_keys=changed_indexes, auto_release=auto_release, info_autosave=info_autosave, verbose=False)
            if verbose:
                logger.info(f'{len(changed_indexes)} items updated.')

    def append(self, item, multiple=False, auto_release=True, info_autosave=True, verbose=True):
        self.insert(len(self), item, multiple=multiple, auto_release=auto_release, info_autosave=info_autosave, verbose=False)
        if verbose:
            logger.success('Append Successfully.')
                


    # def __setitem__(self, key, item):
    #     self._check_key(key)
    #     self.data = self.get_data()
    #     old_v = self.data[key]

    #     self.data[key] = item
    #     if self.load_path is not None:
    #         deep_iter_items()
# def meta_property(func):
#     def setter(self, value):
#         self.meta[func.__name__] = value
#         func(self, value)
#         if self.meta_file_exists:
#             self.save_meta()

#     def getter(self):
#         return self.meta[func.__name__]

#     return property(getter, setter)

# class LooseTree(LooseBase):
#     @meta_property
#     def type(self, value):
#         pass

#     @property
#     def dir(self):
#         return self._dir
    
#     def auto_update_items(self):
#         for k, v in self.item_map.items():
#             if isinstance(v, LooseAtom):
#                 if self.dir is None:
#                     v.save_path = None
#                 else:
#                     v.save_path = os.path.join(self.dir, k)
#             else:
#                 # Is LooseIterable type
#                 v.dir = os.path.join(self.dir, k)

#     @dir.setter
#     def dir(self, new_dir):
#         self._dir = new_dir
#         self.auto_update_items()

#     @property
#     def _default_meta(self):
#         return {
#             'type': self.__class__.__name__
#         }

#     @property
#     def meta_file_exists(self):
#         return self.dir is not None and os.path.isfile(self.get_meta_file_path(self.dir))

#     @staticmethod
#     def get_check_meta_by_tree_dir(path):
#         if not os.path.isdir(path):
#             raise FileNotFoundError(f"The directory of LooseTree doesn't exist: {path}")
#         else:
#             meta_path = LooseTree.get_meta_file_path(path)
#             if not os.path.isfile(meta_path):
#                 raise FileNotFoundError(f"It is not a LooseTree directory ('.meta.json' doesn't exist): {path}")
#             else:
#                 meta = utils.load_json(meta_path)
#                 if type(meta) != dict or 'type' not in meta:
#                     raise ValueError(f'The LooseTree carries illegal metadata: {meta_path}')
                
#                 return meta

#     @classmethod
#     def is_matched_from_path(cls, path):
#         try:
#             meta = cls.get_check_meta_by_tree_dir(path)
#             return meta['type'] == cls.__name__
#         except:
#             return False
        

#     @property
#     def meta(self):
#         return self._meta

#     @meta.setter
#     def meta(self, new_meta):
#         self._meta = new_meta
#         if self.meta_file_exists:
#             self.save_meta()

#     @staticmethod
#     def get_meta_file_path(meta_dir):
#         return os.path.join(meta_dir, '.meta.json')
    
#     def save_meta(self):
#         return utils.save_json(self.get_meta_file_path(self.dir), self._meta)
    
#     def load_meta(self):
#         meta = utils.load_json(self.get_meta_file_path(self.dir))
#         self._meta = meta
#         return meta

#     def __init__(self, dir_path=None, meta=None):
#         self.item_map = dict()
#         self.dir = dir_path
#         self.meta = meta if meta is not None else self._default_meta

#     @staticmethod
#     def _check_value_type(v):
#         assert isinstance(v, LooseTree) or isinstance(v, LooseAtom)

#     def clear(self):
#         utils.remove_all_files_in_dir(self.dir)

#     def _check_dir(self):
#         if self.dir is None:
#             raise Exception('Have not set the root dir yet')
#         elif type(self.dir) != str:
#             raise Exception(f'Wrong root dir type: {type(self.dir)}')
#         elif type(self.dir) == str and not os.path.isdir(os.path.dirname(self.dir)):
#             raise Exception(f"Parent direcotry doesn't exist: {os.path.dirname(self.dir)}")
    
#     def save_items(self):
#         for k, v in self.item_map.items():
#             v.save_to_disk(os.path.join(self.dir, k))

#     def save_to_disk(self, new_dir=None):
#         if new_dir is not None:
#             self.dir = new_dir
#         self._check_dir()
#         if os.path.isdir(self.dir):
#             self.clear()
#         else:
#             os.mkdir(self.dir)  
#         self.save_meta()
#         self.save_items()        

#     @classmethod
#     def load_from_disk(cls, load_dir):
#         assert os.path.isdir(load_dir)
#         root = cls(load_dir)
#         root.load_meta()
#         # need check meta
#         # assert get_class_by_name(root.meta['type']).__name__ == cls.__name__
#         from . import function
#         for path in glob(os.path.join(load_dir, '*')):
#             name = os.path.basename(path)
#             try:
#                 child_class = function.get_class_by_path(path)
#                 child = child_class.load_from_disk(path)
#                 root.item_map[name] = child
#             except:
#                 continue
#         return root

#     def __str__(self):
#         result_dict = dict()
#         result_dict['type'] = self.meta['type']
#         result_dict['dir'] = self.dir
#         result_dict['children'] = dict()
#         for k, v in self.item_map.items():
#             if isinstance(v, LooseAtom):
#                 result_dict['children'][k] = {'type': v.__class__.__name__, 'state': v.state_str, 'data': v.data, 'load_path': v.load_path}
#             else:
#                 result_dict['children'][k] = json.loads(str(v))
#         return json.dumps(result_dict, indent=2)
    
#     @property
#     def full_length(self):
#         return len(self.item_map)

#     def __len__(self):
#         return self.full_length
    
#     def __contains__(self, k):
#         return k in self.item_map
    
#     @staticmethod
#     def _wrap_index(index):
#         return index

#     def __getitem__(self, index):
#         return self.item_map[self._wrap_index(index)]
    
#     def __delitem__(self, index):
#         index = self._wrap_index(index)
#         if self.dir is not None:
#             path = os.path.join(self.dir, index)
#             if os.path.exists(path):
#                 if os.path.isdir(path):
#                     shutil.rmtree(path)
#                 elif os.path.isfile(path):
#                     os.remove(path)

#         del self.item_map[index]

#     def __setitem__(self, k, v):
#         k = self._wrap_index(k)

#         self._check_value_type(v)

#         if k in self:
#             del self[k]

#         if self.dir is not None:
#             path = os.path.join(self.dir, k)
#             if isinstance(v, LooseAtom):
#                 v.save_path = path
#             else:
#                 v.dir = path
                
#             if os.path.isdir(self.dir):
#                 v.save_to_disk()

#         self.item_map[k] = v

#     def get_key_by_item(self, item):
#         for k, v in self.item_map.items():
#             if v == item:
#                 return k
#         else:
#             raise Exception(f'Item cannnot find: {item}')


# class LooseDict(LooseTree):
#     pass

# # test_root_dir = 'D:/documents/AcademicDocuments/customed_python_pkgs/loosely/test_loose_dict'
# # root = LooseDict.load_from_disk(test_root_dir)
# # root['child2']['test.json'] = LooseAtom(555)
# # child2 = LooseDict()
# # root['child2'] = child2
# # print(root)

# class LooseList(LooseTree):
#     @property
#     def elements(self):
#         element_items = [{'index': int(k), 'item': item} for k, item in self.item_map.items() if k.isdecimal()]
#         element_items.sort(key=lambda d: d['index'])
#         return [d['item'] for d in element_items]

#     def check_elements(self):
#         for i, e in enumerate(self.elements):
#             k = self.get_key_by_item(e)
#             if i != int(k):
#                 raise Exception(f"Indexes not align, index={i} but item's name= {k}")

#     def __init__(self, dir_path=None, meta=None):
#         super().__init__(dir_path, meta)

#     def __len__(self):
#         return len(self.elements)

#     def insert(self, index, new_v):
#         after_elements = self.elements[index:]
#         after_elements.reverse()
#         for after_e in after_elements:
#             index = int(self.get_key_by_item(after_e))
#             self[index+1] = after_e
#         self[index] = new_v

#     def append(self, new_v):
#         self.insert(len(self), new_v)

#     def update(self, index, new_v):
#         self[index] = new_v

#     def pop(self, index=-1):
#         e = self.elements[index]
#         after_elements = self.elements[index:]
#         del self[self.get_key_by_item(after_elements[0])]
#         for after_e in after_elements[1:]:
#             i = int(self.get_key_by_item(after_e))
#             self[i-1] = after_e
#             del self[i]
#         return e

#     @staticmethod
#     def _wrap_index(index):
#         if type(index) == int:
#             index = str(index)
#         return index


# test_root_dir = 'D:/documents/AcademicDocuments/customed_python_pkgs/loosely/test_loose_dict'
# root = LooseDict.load_from_disk(test_root_dir)
# root.type = 'LooseDict'
# print(root.type)
# print(root.type)
# ele = LooseDict()
# ele['teslll.json'] = LooseAtom({'f': None})
# root['child3'].append(ele)
# print([1,2,3][-1:])
# root['child3'][0] = ele
# print(root['child3'].pop(0))
# root['child3'].check_elements()
# for e in root['child3'].elements:
#     print(e)
# print()
# import sys
# print(sys.modules[__file__])
# print(utils.get_all_classes_in_module(__name__))
    
# class LooseArrayElement(LooseAtom):
#     def auto_file_update(func):
#         def getter(self):
#             return getattr(self, '_'+func.__name__)
#         def setter(self, value):
#             try:
#                 old_path = self.file_path
#                 old_exists = old_path is not None and os.path.isfile(old_path)
#             except:
#                 setattr(self, '_'+func.__name__, value)
#                 return
            
#             if old_exists:
#                 os.remove(old_path)
#             setattr(self, '_'+func.__name__, value)
#             self.save_path = self.file_path
#             if self._check() and old_exists:
#                 self.save_to_disk()
#         return property(getter, setter)

#     @auto_file_update
#     def dir(self):
#         pass

#     @auto_file_update
#     def index(self):
#         pass

#     def __init__(self, data, save_dir=None, index=None):
#         self.dir = save_dir
#         self.index = index
#         super().__init__(data, self.file_path)

#     def _check(self):
#         if self.dir is None and self.index is None:
#             return False
#         elif self.dir is not None and self.index is not None and os.path.isdir(self.dir) and type(self.index) == int:
#             return True
#         else:
#             raise Exception(f'LooseArrayElement checking failed: dir={self.dir}, index={self.index}')

#     @staticmethod
#     def get_file_name(index):
#         return f'{index}.json'
    
#     @property
#     def file_name(self):
#         assert self.index is None or type(self.index) == int
#         return self.get_file_name(self.index) if self.index is not None else None

#     @staticmethod
#     def get_file_path(file_dir, index):
#         return os.path.join(file_dir, LooseArrayElement.get_file_name(index))

#     @property
#     def file_path(self):
#         return self.get_file_path(self.dir, self.index) if self._check() else None
    
#     @classmethod
#     def load_from_disk(cls, save_dir, index):
#         arr = cls.__base__.load_from_disk(cls.get_file_path(save_dir, index))
#         return cls(arr.data, save_dir, index)
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
# test_root_dir = 'D:/documents/AcademicDocuments/customed_python_pkgs/loosely/test_loose_dict'
# root = LooseDict.load_from_disk(test_root_dir)

# sub_tree_items = list(deep_iter_items(data, lambda v: isinstance(v, LooseTree), iter_get_data=False))

        # 筛选item,使得各tree没有交集，且子父tree同时存在时只取父tree
        # filtered_tree_items = []
        # for item in sub_tree_items:
        #     for i, item2 in enumerate(filtered_tree_items):
        #         if is_parent_trace(item[1], item2[1]):
        #             filtered_tree_items[i] = None
        #         elif is_parent_trace(item2[1], item[1]):
        #             break
        #     else:
        #         filtered_tree_items.append(item)
        # while None in filtered_tree_items:
        #     filtered_tree_items.remove(None)
        # print([item[1] for item in filtered_tree_items])