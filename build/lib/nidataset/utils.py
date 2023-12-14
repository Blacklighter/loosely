# from zipfile import ZipFile, ZIP_BZIP2, ZIP_STORED
import os
import pyarrow as pa
import pandas as pd
import numpy as np
from collections.abc import Iterable
from uuid import uuid1
import json, random, re, time
from tqdm import tqdm
import shutil, inspect
from glob import glob
import processor

EXTERNAL_TAG = '<external>'
EXTERNAL_PATTERN = re.compile(rf'^{EXTERNAL_TAG}(.+)')
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

def get_row_path(root, index):
    return os.path.join(root, 'table', f'{index}.row')

def get_row_file_path(root, index, file_name):
    return os.path.join(get_row_path(root, index), file_name)

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
    pipeline = processor.Pipeline(pipeline_dir=get_pipeline_dir(path))
    pipeline.save()

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
    row_path = get_row_path(root, index)
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