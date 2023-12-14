# from zipfile import ZipFile, ZIP_BZIP2, ZIP_STORED
import os
# import pyarrow as pa
# import pandas as pd
import numpy as np
from collections.abc import Iterable
from uuid import uuid1
import json, random, re, time
from tqdm import tqdm
import shutil, inspect
from glob import glob
import datetime, sys, re
from uuid import uuid1
from pathlib import Path
from collections import Callable

EXTERNAL_TAG = '<external>'
EXTERNAL_PATTERN = re.compile(rf'^{EXTERNAL_TAG}(.+)')

def get_example_path(*paths):
    return os.path.abspath(os.path.join(__file__, '..', '..', 'examples', *paths))

def is_callable(obj):
    return isinstance(obj, Callable)
def get_keys(data):
    keys = []
    if hasattr(data, 'keys') and is_callable(data.keys):
        keys = data.keys()
    elif type(data) in [tuple, list]:
        keys = range(len(data))
    for key in keys:
        yield key

def get_values(data):
    for key in get_keys(data):
        yield data[key]

def get_items(data):
    for key in get_keys(data):
        yield key, data[key]

def find_deep_index(d, filter_func=None, search_get_data=True):
    filter_func = filter_func if filter_func is not None else lambda d2: True
    def _func(d2, trace, trace_list):
        if filter_func(d2):
            trace_list.append(trace)
        t = type(d2)
        keys = []
        if t in [dict, tuple, list]:
            keys = get_keys(d2)
        elif search_get_data and hasattr(d2, 'get_data'):
            data = d2.get_data()
            keys = get_keys(data)
        for k in keys:
            trace_copy = json.loads(json.dumps(trace))
            trace_copy.append(k)
            _func(d2[k], trace_copy, trace_list)

    index = []
    t = type(d)
    keys = []
    if t in [dict, tuple, list]:
        keys = get_keys(d)
    elif search_get_data and hasattr(d, 'get_data'):
        data = d.get_data()
        keys = get_keys(data)
    for k in keys:
        _func(d[k], [k], index)
    return index

def get_deep_item(d, trace):
    for key in trace:
        d = d[key]
    return d

def set_deep_item(d, trace, new_item):
    for key in trace[:-1]:
        d = d[key]
    d[trace[-1]] = new_item

def deep_iter_items(d, filter_func=None, iter_get_data=True):
    traces = find_deep_index(d, filter_func, search_get_data=iter_get_data)
    for trace in traces:
        parent_d = d
        for key in trace[:-1]:
            parent_d = parent_d[key]
        yield parent_d[trace[-1]], trace, parent_d

def is_leaf_obj(obj):
    t = type(obj)
    if t in [int, float, complex, str, bool, np.ndarray, None.__class__]:
        return True
    else:
        if t in [tuple, list, dict]:
            return len(obj) == 0
        else:
            raise TypeError(f"Cannnot identify whether it is a leaf type: '{t}'")

def is_parent_trace(trace1, trace2):
    # trace1是否时trace2的父trace
    if len(trace1) >= len(trace2):
        return False
    for i, key in enumerate(trace1):
        if key != trace2[i]:
            return False
    return True

def trace_all_leaves(d, extra_filter_func=None):
    for v, trace, parent in deep_iter_items(d, extra_filter_func):
        if is_leaf_obj(v):
            yield v, trace, parent


def simple_deep_copy(d):
    return json.loads(json.dumps(d))

def deep_copy(d, extra_copy_tuples=[]):
    copy_tuples = [(np.ndarray, np.copy)]
    copy_tuples.extend(extra_copy_tuples)

    def copy_leaf(leaf):
        copy_funcs = [copy_tuple[1] for copy_tuple in copy_tuples if type(leaf) == copy_tuple[0]]
        if len(copy_funcs) > 0:
            return copy_funcs[0](leaf)
        else:
            return simple_deep_copy(leaf)

    def infer_parent_type_by_key(key):
        key_type = type(key)
        if key_type == int:
            return list
        elif key_type == str:
            return dict
        raise TypeError(f'Unknown index type: {key}')

    def _get_parent_by_key(parent, key):
        infered_parent_type = infer_parent_type_by_key(key)
        if type(parent) != infered_parent_type:
            parent = infered_parent_type()
        if infered_parent_type == list:
            if key >= len(parent):
                parent.extend([None]*(key-len(parent)+1))
        return parent

    def _copy_leaf_from_trace(copy_d, trace, leaf_v):
        leaf_v = copy_leaf(leaf_v)
        if len(trace) == 0:
            return copy_d
        else:
            d = _get_parent_by_key(copy_d, trace[0])
            parent = d
            for i, key in enumerate(trace[:-1]):
                next_key = trace[i+1]
                try:
                    next_parent = parent[key]
                except:
                    next_parent = None
                parent[key] = _get_parent_by_key(next_parent, next_key)
                parent = parent[key]
            parent[trace[-1]] = leaf_v
            return d

    if is_leaf_obj(d):
        return copy_leaf(d)
    else:
        copy_d = None
        for i, (v, trace, parent) in enumerate(trace_all_leaves(d)):
            copy_d = _copy_leaf_from_trace(copy_d,trace, v)
        return copy_d



def deep_map(d, map_func=None, filter_func=None):
    map_func = map_func if map_func is not None else lambda src: simple_deep_copy(src)
    copy_d = type(d)()
    # print(d)
    # print(copy_d, 1)
    traces = find_deep_index(d, filter_func)
    # print(copy_d, 2)
    for trace in traces:
        tmp_d = copy_d
        tmp_src_d = d
        # print(trace, copy_d)
        for key in trace[:-1]:
            # print(copy_d, key)
            try:
                if type(tmp_d[key]) != type(tmp_src_d[key]):
                    raise
            except:
                empty_item = type(tmp_src_d[key])()
                if type(key) == int and len(tmp_d) <= key:
                    tmp_d.extend([None]*(key-len(tmp_d)))
                    tmp_d.append(empty_item)
                else:
                    tmp_d[key] = empty_item
                
            tmp_d = tmp_d[key]
            tmp_src_d = tmp_src_d[key]
        if len(trace) > 0:
            last_index = trace[-1]
            # print(trace)
            new_item = map_func(tmp_src_d[last_index])
            # print(trace, tmp_d, tmp_src_d)
            if type(last_index) == int:
                if last_index >= len(tmp_d):
                    if last_index > len(tmp_d):
                        tmp_d.extend([None]*(last_index-len(tmp_d)))
                    tmp_d.append(new_item)
                else:
                    tmp_d[last_index] = new_item
            else:
                tmp_d[last_index] = new_item
            # print(trace, tmp_d, tmp_src_d, copy_d)
            
    return copy_d

TRACE_DIVIDER = '-'
LOOSELY_STR_RE = re.compile(r"<loosely-(\S+) '(.*)' \|(.*)\|>")

def is_loosely_str(loosely_str, restrict_prefix='*'):
    r = (type(loosely_str) == str) and (LOOSELY_STR_RE.match(loosely_str) is not None)
    if not r or (restrict_prefix == '*'):
        return r
    else:
        alias, file_stem, extra_str = LOOSELY_STR_RE.match(loosely_str).groups()
        splits = alias.split('_')
        if len(splits) > 0:
            return splits[0] == restrict_prefix
        else:
            return False

def remove_by_path(path):
    path = Path(path)
    if path.exists():
        if path.is_file():
            os.remove(str(path.resolve()))
        else:
            shutil.rmtree(str(path.resolve()))
        return True
    else:
        return False

def extract_loosely_str(loosely_str):
    if is_loosely_str(loosely_str):
        alias, file_stem, extra_str = LOOSELY_STR_RE.match(loosely_str).groups()
        t = get_cls_from_alias(alias)
        return {
            'type': t,
            'file_name': file_stem+t.EXT,
            'extra': t.extract_extra_from_str(extra_str)
        }
    else:
        return None


def get_loosely_name_by_trace(trace):
    return TRACE_DIVIDER.join([str(p) for p in trace])

# def get_loosely_item_str(prefix, stem, suffix=''):
#     if type(stem) == list or type(stem) == tuple:
#         stem = TRACE_DIVIDER.join([str(item) for item in stem])
#     return f"<loosely-{prefix} '{stem}{suffix}'>"

def get_cls_from_alias(alias):
    from loosely import atom, tree, dataset
    m = {
        'atom_dict': atom.LooseAtomDict,
        'atom_arr': atom.LooseAtomArray,
        'atom_obj': atom.LooseAtomObj,
        'atom_func': atom.LooseAtomFunc,
        'tree_dict': tree.LooseDict,
        'tree_arr': tree.LooseArray,
        'tree_dataset': dataset.LooseDataset
    }

    if alias not in list(m.keys()):
        raise NameError(f'Unknown loosely class alias: {alias}')
    else:
        return m[alias]
    
def generate_id():
    return str(uuid1())
# def deep_copy(src):
#     # def get_copy_by_trace(d, trace):
#     #     parent_d = d
#     #     for key in trace:
#     #         parent_d = parent_d[key]
#     #     if type(parent_d) == np.ndarray:
#     #         return parent_d.copy()
#     #     else:
#     #         return json.loads(json.dumps(parent_d))

#     def _deep_copy(src_d, copy_d, trace):
#         p_src_d = src_d
#         p_copy_d = copy_d
#         for key in trace[:-1]:
#             p_src_d = p_src_d[key]
#             p_copy_d = p_copy_d[key]
#         v = p_src_d[trace[-1]]
#         if type(v) == np.ndarray:
#              p_copy_d[trace[-1]] = v.copy()
#         elif isinstance(v, Iterable):
#             for k in v:
#                 trace_copy = json.loads(json.dumps(trace))
#                 trace_copy.append(k)
#                 _deep_copy(src_d, copy_d, trace_copy)
#         else:
#             p_copy_d[trace[-1]] = json.loads(json.dumps(v))

def create_external_path_str(path):
    return EXTERNAL_TAG + path

def get_external_name(s):
    groups = EXTERNAL_PATTERN.match(s)
    return groups[1] if groups else None

def is_external_path_str(s):
    return get_external_name(s) is not None

def get_ext(path):
    return os.path.splitext(path)[-1]

def check_ext(path):
    assert get_ext(path) == '.nids'

def read(path):
    check_ext(path)
    return path

def get_row_dir(root, index):
    return os.path.join(root, 'table', f'{index}.row')

def get_row_index_by_dir(row_dir):
    return int(os.path.basename(row_dir).split('.')[0])

def get_row_file_path(root, index, file_name):
    return os.path.join(get_row_dir(root, index), file_name)

def save_json(path, obj, indent=2):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=indent)

def load_json(path):
    f = open(path)
    data = json.load(f)
    f.close()
    return data


INFO_FILE_NAME = 'info.json'
def get_info_abs_path(root, index):
    return get_row_file_path(root, index, INFO_FILE_NAME)

# COMPRESSION_MAP = {
#     'no_compression': (ZIP_STORED, None),
#     'bz2_max': (ZIP_BZIP2, 9)
# }
def get_header_path(root):
    HEADER_FILE_NAME = 'header.json'
    return os.path.join(root, HEADER_FILE_NAME)

def get_table_dir(root):
    return os.path.join(root, 'table')

def get_extra_dir(root):
    return os.path.join(root, 'extra')

def get_pipeline_dir(root):
    return os.path.join(root, 'pipeline')

def create_empty_file(path, name=None, use_pipeline=True):
    # check if the dataset path is right for requirement
    check_ext(path)
    # clear the target dataset directory if it is not empty
    if os.path.isdir(path):
        shutil.rmtree(path)

    # create directories for initialization
    dirs_to_create = [path, get_table_dir(path), get_extra_dir(path), get_pipeline_dir(path)]
    for dir_to_create in dirs_to_create:
        os.mkdir(dir_to_create)

    # initiate header files
    header_dict = {
        'name': name if name is not None else os.path.basename(path).split('.')[0],
        'use_pipeline': use_pipeline
    }
    save_json(get_header_path(path), header_dict)

    # initiate Pipeline files
    # pipeline = processor.Pipeline(pipeline_dir=get_pipeline_dir(path))
    # pipeline.save()

    # initiate filterd_indexes
    

    return path

def create_sub_files(dir_path, d):
    for k,v in d.items(): 
        # print(k, v, type(v))
        if (type(v) == list or type(v) == tuple):
            tmp = np.asarray(v)
            if np.issubdtype(tmp.dtype, np.number):
                v = tmp
        # print(k, v, type(v))
        if isinstance(v, np.ndarray):
            np_file_name = f'{uuid1()}.npy'
            np_file_path = os.path.join(dir_path, np_file_name)
            np.save(np_file_path, v)
            d[k] = create_external_path_str(np_file_name)
    else:
        save_json(os.path.join(dir_path, INFO_FILE_NAME), d)


def create_row(root, index, d, overwrite=False):
    row_path = get_row_dir(root, index)
    try:
        os.mkdir(row_path)
        create_sub_files(row_path, d)
    except Exception as e:
        if overwrite:
            shutil.rmtree(row_path)
            os.mkdir(row_path)
            create_sub_files(row_path, d)
        else:
            raise e
    finally:
        return row_path

def open_external_file(path):
    ext = get_ext(path)
    SUPPORT_EXTERNAL_FILE = ['.npy']
    assert ext in SUPPORT_EXTERNAL_FILE
    
    if ext == '.npy':
        result = np.load(path)

    return result


def is_number(v):
    return np.issubdtype(type(v), np.number) 

def is_all_numeric(l):
    return all([is_number(v) for v in l])

def remove_all_files_in_dir(dir_path):
    for path in glob(os.path.join(dir_path, '*')):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

def get_func_param_num(func):
    return len(inspect.signature(func).parameters)

def get_current_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_all_classes_in_module(module, filter_func=None):
    classes = []
    for name, class_ in inspect.getmembers(module, inspect.isclass):
        if callable(filter_func) and filter_func(class_) or not callable(filter_func):
            classes.append(class_)
    return classes

def deep_find(d, func, find_all=False, with_indexes_return=False):
    results = []
    def _find(data, cur_indexes=[]):
        if isinstance(data, Iterable) and type(data) != str:
            if type(data) == dict:
                d_iter = data.items()
            else:
                d_iter = enumerate(data)
            for k, v in d_iter:
                input_indexes = cur_indexes[:] + [k] if with_indexes_return else None
                if _find(v, input_indexes):
                    return True
        if func(data):
            results.append((data, cur_indexes) if with_indexes_return else data)
            if not find_all:
                return True
        return False

    _find(d)
    
    # print(results)
    if len(results) > 0:
        return results if find_all else results[0]
    else:
        return None
    
def deep_include(d, func):
    return deep_find(d, func) is not None

def get_loose_class_by_load_path(load_path):
    from loosely.atom import LooseAtomDict, LooseAtomArray
    from loosely.tree import LooseDict
    classes = [LooseAtomDict, LooseAtomArray, LooseDict]
    
    # print(LooseDict.is_matched_load_path(load_path))
    c = list(filter(lambda c2: c2.is_matched_load_path(load_path), classes))
    if len(c) > 0:
        return c[0]
    else:
        return None


if __name__ == '__main__':
    pass
    # d = {
    #     'a': 21331,
    #     'asdfa': {
    #         'b': 'fff',
    #         'c': 123,
    #         'ff': 11.5
    #     },
    #     'd': [
    #         'fa',
    #         0,
    #         101
    #     ]
    # }

    # print(deep_find(d, lambda item: type(item) == int and item > 100000, find_all=True, with_indexes_return=True))
    # print(123)
