import numpy as np
import os, utils, json
from uuid import uuid1
from collections.abc import Iterable

class LooseAtom:
    def __init__(self, data=None, save_path=None, load_path=None):
        # keep in memory
        self.data = data
        self.save_path = save_path
        self.load_path = load_path

    @property
    def state(self):
        try:
            self._check_save_path()
            return f'Ready to save at {self.save_path}' if not os.path.isfile(self.save_path) else f'Saved at {self.save_path}'
        except:
            return 'In Memory'
        
    def _check_save_path(self):
        if self.save_path is None:
            raise Exception('Save path have not been set.')
        elif type(self.save_path) != str:
            raise Exception(f'Save path is a not string: {type(self.save_path)}')
        elif not os.path.isdir(os.path.dirname(self.save_path)):
            raise Exception(f"The parent dierectory doesn't exist: {os.path.dirname(self.save_path)}")
        
    def save_to_disk(self, save_path=None):
        if save_path is not None:
            self.save_path = save_path
        self._check_save_path()
        utils.save_json(self.save_path, self.data)
        self.release_data()
        self.load_path = self.save_path
        
    @classmethod
    def load_from_disk(cls, load_path=None):
        data = utils.load_json(load_path)
        return cls(data, load_path, load_path)
    
    def load_data(self, with_attr_update=True):
        assert self.load_path is not None and os.path.isfile(self.load_path)
        data = utils.load_json(self.load_path)
        if with_attr_update:
            self.data = data
        return data

    def release_data(self):
        del self.data
        self.data = None

    def __str__(self):
        return json.dumps({
            'state': self.state,
            'data': self.data
        }, indent=2)
    
    def __getitem__(self, index):
        data = self.data if self.data is not None else self.load_data(with_attr_update=False)
        return data[index]
    
    def __setitem__(self, index, new_v):
        data = self.data if self.data is not None else self.load_data(with_attr_update=False)
        data[index] = new_v
        if self.load_path is not None and (self.save_path is None or self.save_path == self.load_path):
            self.data = data
            self.save_to_disk()
# x = {
#     'fa': 1,
#     'fff': None
# }
# save_path = 'D:/documents/AcademicDocuments/customed_python_pkgs/loosely/test.json'
# # print(LooseAtom._check_obj(set([1,2,3])))
# atom = LooseAtom(x, save_path)
# atom.save_to_disk()
# atom = LooseAtom.load_from_disk(save_path)
# print(atom)
# np.savetxt(save_path, np.array([9, '123', None]))
# a = {'fda': 213}
# for x in a:
#     print(x)