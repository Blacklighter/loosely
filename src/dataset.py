import json
from . import utils
from typing import Any
from zipfile import ZipFile
import numpy as np
import json, re, pandas as pd
import os, functools, pickle
from time import perf_counter
from multiprocessing import Pool
import warnings, random, shutil
from loosely.tree import LooseDict, LooseAtomDict, LooseArray
from pathlib import Path
from loguru import logger
# from tqdm import tqdm
# from uuid import uuid1
# from glob import glob
# from collections.abc import Iterable
# from torch.utils.data import Dataset, DataLoader
# from .process import Wrapper, Filter, Pipeline
# from .tree import LooseDict, meta_property, LooseList
# from .atom import LooseJsonAtom, LooseBundleAtom
# import torch

class LooseDataset(LooseDict):
    EXT = '.dst'
    INFO_FILE_NAME = '.header.json'
    SUPPORT_INPUT_DATA_TYPE = LooseAtomDict.SUPPORT_INPUT_DATA_TYPE
    CLS_ALIAS = 'tree_dataset'


    def __init__(self, meta={}, table=[], extra={}):
        data = {
            'meta': LooseDict(meta),
            'table': LooseArray(table),
            'extra': LooseDict(extra)
        }
        super().__init__(data)

    @property
    def meta(self):
        return self['meta']
    
    @property
    def table(self):
        return self['table']

    @property
    def extra(self):
        return self['extra']


    @classmethod
    def _create_iter_step(cls, obj, k, v):
        obj.table.append(v, verbose=False)

    @classmethod
    def create_from_iter(cls, dir_path, iterator, length=None, meta={}):
        if Path(dir_path).exists():
            logger.warning(f'{cls.__name__} exists, ready to overwrite...')
        def it2():
            i = 0
            for item in iterator:
                yield i, item
                i += 1
        return super().create_from_iter(dir_path, it2(), length, meta=meta)
# class LooseDataset(LooseDict):
#     # @meta_property
#     # def name(self, new_value):
#     #     pass

#     @meta_property
#     def use_pipeline(self, new_value):
#         pass

#     @property
#     def table(self):
#         return self['table']
    
#     @property
#     def extra(self):
#         return self['extra']
    
#     @property
#     def pipeline(self):
#         return self['pipeline']

#     # def __init__(self, nids_dir=None, meta=None):
#     #     super().__init__(nids_dir, meta)
        


#     def init_sub_dirs(self):
#         self['table'] = LooseList()
#         self['extra'] = LooseDict()
#         self['pipeline'] = LooseDict()
#     # def __init__(self, nids_dir):
#     #     assert os.path.exists(nids_dir)
#     #     self.dir = nids_dir
#     #     self.reload()

#     # @property
#     # def extra(self):
#     #     return self._get_data_from_dir(self.extra_dir)
    
#     # @extra.setter
#     # def extra(self, new_extra):
#     #     assert type(new_extra) == dict or new_extra is None
#     #     extra_dir = self.extra_dir
#     #     utils.remove_all_files_in_dir(extra_dir)
#     #     if new_extra is not None:
#     #         utils.create_sub_files(extra_dir, new_extra)
            
#     # @property
#     # def header_path(self):
#     #     return utils.get_header_path(self.dir)

#     # @property
#     # def table_dir(self):
#     #     return utils.get_table_dir(self.dir)
    
#     # @property
#     # def extra_dir(self):
#     #     return utils.get_extra_dir(self.dir)
    
#     # @property
#     # def pipeline_dir(self):
#     #     return utils.get_pipeline_dir(self.dir)
    
#     # @property
#     # def has_filter(self):
#     #     return any([isinstance(processor, Filter) for processor in self.pipeline.processors])

#     @classmethod
#     def create_from_iter(cls, nids_dir, it, name=None, pbar_title='Creating nidataset', it_len=None, extra_items=[], pipeline_items=[]):
#         start_t = perf_counter()
#         # root = utils.create_empty_file(nids_dir, name)
#         dataset = cls(nids_dir)
#         dataset.init_sub_dirs()
#         dataset.name = name
#         dataset.save_to_disk()
#         pbar = tqdm(it if it_len is None else range(it_len))

#         for i, d in enumerate(pbar):
#             if it_len is not None:
#                 d = next(it)
#             # contain list data
#             row = LooseDict()
#             if utils.deep_include(d, lambda item: type(item) in [list, tuple, np.ndarray]):
#                 atom = LooseBundleAtom(d)
#             else:
#                 atom = LooseJsonAtom(d)
#             row[f'data{atom.get_support_ext()}'] = atom
#             dataset.table.append(row)
#             # row_path = utils.create_row(root, i, d)
#             pbar.set_description(f"{pbar_title} '{i}'")
#         print(f'Time consuming for create: {round(perf_counter()-start_t, 2)}s')
#         # ds = cls(nids_dir)
#         if len(extra_items) > 0:
#             for k, extra_item in extra_items:
#                 dataset.extra[k] = extra_item

#         if len(pipeline_items) > 0:
#             for k, pipeline_item in pipeline_items:
#                 dataset.pipeline[k] = pipeline_item
#         # if len(extra_processors) > 0:
#         #     for processor in extra_processors:
#         #         ds.pipeline.append(processor)
#         #     ds.pipeline.save()
#         return dataset

#     @classmethod
#     def create_example(cls, path, n=1000):
#         def data_iter():
#             for i in range(n):
#                 yield {
#                     'a': np.random.rand(121, 145, 121),
#                     'b': [random.randint(-10,10), random.randint(-10,10), random.randint(-10,10)],
#                     'c': str(uuid1()),
#                     'd': random.random()*100,
#                     'ff': False if random.random() > 0.5 else True,
#                     'dfa': None
#                 }
#         ds = cls.create_from_iter(path, data_iter(), it_len=n)
#         # Pipeline.create_example(ds.pipeline_dir)  
#         return ds

#     def reload_passed_index(self):
#         row_paths = self.get_row_paths(with_filter=False)
#         if self.has_filter:
#             pass
#         else:
#             self.passed_index = []

#     def reload(self):
#         self.header = utils.load_json(self.header_path)
#         self.pipeline = Pipeline.load_from_dir(self.pipeline_dir)
#         # self

#     # @property
#     # def file(self):
#     #     return read(self.dir)
#     def get_row_paths(self, with_filter=True):
#         return [path for path in glob(os.path.join(self.table_dir, '*.row')) if os.path.isdir(path)]


#     def __len__(self):
#         return len(self.passed_indexes)

#     def get_unfilterd_len(self):
#         return len(self.get_row_paths(with_filter=False))

    # def __getitem__(self, index):
    #     if isinstance(index, int):
    #         return self._get_row(index)
    #     elif isinstance(index, slice):
    #         start, stop, step = index.indices(len(self))
    #         indexes = list(range(start, stop, step))
    #         return [self[i] for i in indexes]
    #     else:
    #         raise Exception(f'Index Error of LooseDataset: {index}')
        
    # def __setitem__(self, index, new_d):
    #     t = type(index)
    #     SUPPORT_TYPE = [int, list, tuple]
    #     assert t in SUPPORT_TYPE
    #     if t == int:
    #         self._check_index(index, (0, len(self)))
    #         self.update_row(index, new_d)
    #     elif t == list or t == tuple:
    #         assert all([type(item) == int for item in index])
    #         assert isinstance(new_d, Iterable) and len(index) == len(new_d)
    #         for i, item in zip(index, new_d):
    #             self[i] = item

#     @staticmethod
#     def _get_data_from_dir(data_dir):
#         info_path = os.path.join(data_dir, utils.INFO_FILE_NAME)
#         if not os.path.exists(info_path):
#             return None
#         d = utils.load_json(info_path)
#         for k, v in d.items():
#             if type(v) == str and utils.is_external_path_str(v):
#                 external_path = os.path.join(data_dir, utils.get_external_name(v))
#                 d[k] = utils.open_external_file(external_path)
#         return d

#     def _get_row(self, index):
#         row_path = utils.get_row_dir(self.dir, index)
#         data = self._get_data_from_dir(row_path)
#         return data

#     # def __str__(self):
#     #     return json.dumps({
#     #         'path': self.dir,
#     #         'len': len(self)
#     #     },indent=2)

#     @staticmethod
#     def _check_index(index, span):
#         assert index >= span[0] and index < span[1]
    
#     def insert_row(self, index, d):
#         self._check_index(index, (0, len(self)+1))
#         tail_indexes = [i for i in range(index, len(self))]
#         if len(tail_indexes) > 0:
#             tail_indexes.reverse()
#             for tail_index in tail_indexes:
#                 os.rename(utils.get_row_dir(self.dir, tail_index), utils.get_row_dir(self.dir, tail_index+1))
#         utils.create_row(self.dir, index, d)

#     def append_row(self, d):
#         self.insert(len(self), d)

#     def update_row(self, index, new_d):
#         utils.create_row(self.dir, index, new_d, overwrite=True)

#     def partial_update_row(self, index, update_map):
#         d = self[index]
#         d.update(update_map)
#         self.update_row(index, d)

#     def remove_row(self, index):
#         self._check_index(index, (0, len(self)))
#         old_len = len(self)
#         shutil.rmtree(utils.get_row_dir(self.dir, index))
#         for tail_index in [i for i in range(index+1, old_len)]:
#             os.rename(utils.get_row_dir(self.dir, tail_index), utils.get_row_dir(self.dir, tail_index-1))
            

#     def clear_table(self):
#         for path in self.get_row_paths():
#             shutil.rmtree(path)
    
#     def apply_map(self, func, save_path, filter_func=None, name=None,  copy_extra=True, new_extra=None):
#         def map_iter():
#             for i in range(len(self)):
#                 d = self[i]
#                 if filter_func is None or filter_func(d, i, self):
#                     yield func(d, i, self)
#                 else:
#                     continue

#         it = map_iter()

#         if copy_extra:
#             new_extra = self.extra
#         new_ds = self.create_from_iter(save_path, it, name, pbar_title='New nidataset mapping', 
#                                        it_len=len(self) if filter_func is None else None, extra=new_extra)
#         if filter_func is not None:
#             print(f'{len(self) - len(new_ds)} rows were filterd, {len(new_ds)} were left.')
#         return new_ds

#     def to_torch_dataset(self, get_item_func=None, len_func=None, init_func=None):
#         def _get_item_func(index, nids):
#             return nids[index]
#         get_item_func = get_item_func or _get_item_func

#         def _len_func(nids):
#             return len(nids)
#         len_func = len_func or _len_func

#         def _init_func(nids):
#             return None
#         init_func = init_func or _init_func

#         class TorchDataset(Dataset):
#             def __init__(self2):
#                 init_func(self)

#             def __getitem__(self2, index):
#                 return get_item_func(index, self)

#             def __len__(self2):
#                 return len_func(self)
            
#         return TorchDataset()
    
#     def to_torch_dataloader(self, batch_size, shuffle=False, get_item_func=None, len_func=None, init_func=None):
#         return DataLoader(self.to_torch_dataset(get_item_func, len_func, init_func), batch_size=batch_size, shuffle=shuffle)

# if __name__ == '__main__':
#     # pass
#     dataset_path = 'D:/documents/AcademicDocuments/customed_python_pkgs/loosely/example_dataset3.nids'
    # dataset = LooseDataset.load_from_disk(dataset_path)
    # print(len(dataset.table))
    # dataset = LooseDataset.create_example(dataset_path, n=100)

    # dataset = LooseDataset.load_from_disk(dataset_path)
    # dataset = LooseDataset(dataset_path)
    # dataset.save_to_disk()
    # print(dataset)
    # print(dataset.use_pipeline)
    # dataset.use_pipeline = False
    # print(dataset.use_pipeline)

    # print(dataset.extra)
    # arr = np.array([[2], [2.3], [True], ['fdafa']])
    # print(arr, arr.dtype)
    # print(dataset[3])
    # print(dataset.extra)
    # dataloader = dataset.to_torch_dataloader(8)
    # print(dataloader)
    # torch_ds = dataset.to_torch_dataset(get_item_func=lambda index, nids: nids[index]['c'])
    # print(torch_ds[1])
    # dataset.apply_map(lambda d, i, ds: {'d':d['d']},
    #                   filter_func=lambda d,i,ds: i%2==0,
    #                   save_path='D:/documents/AcademicDocuments/customed_python_pkgs/neuro-dataset/example_dataset2.nids')
    # dataset = LooseDataset(dataset_path))
    # data = [round(d['d'], 2) for d in dataset[:100]]
    # print(np.array(data))
    # print(f'Consuming {perf_counter()-start_t}s')
# print(f.namelist())
# f2 = f.open('table/0/col1.npy', 'a')
# print(np.load(f2))