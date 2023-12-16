from LooseAtom import LooseAtom
import utils, os

class LooseNamedObj:
    HEADER_FILE_NAME = 'header.json'
    def header_property(func):
        def setter(self, value):
            self.header.data[func.__name__] = value
            self.save_header()
            func(self, value)

        def getter(self):
            return self.header.data[func.__name__]
        
        return property(getter, setter)

    def __init__(self, dir_path=None, name=None):
        # keep in memory
        self.header = self.get_default_header()
        if name is not None:
            self.name = name
        self.dir = dir_path

    @property
    def dir(self):
        return self._dir
    
    @dir.setter
    def dir(self, new_dir):
        self._dir = new_dir
        if self.dir is not None and self.name is None:
            self.header.data['name'] = os.path.basename(self.dir)
        if self.dir is not None:
            self.header.save_path = self.header_path

    @property
    def header_path(self):
        return os.path.join(self.dir, self.HEADER_FILE_NAME)
    
    @header_property
    def name(self, new_name):
        pass

    @property
    def lately_update_time(self):
        return self.header.data['lately_update_time']
        
    @staticmethod
    def get_default_header():
        return LooseAtom({
            'name': None,
            'lately_update_time': utils.get_current_time_str()
        })

    def refresh_header_lately_update_time(self):
        self.header.data['lately_update_time'] = utils.get_current_time_str()

    def _check_dir(self):
        if self.dir is None:
            raise Exception('Have not set the root dir yet')
        elif type(self.dir) != str:
            raise Exception(f'Wrong root dir type: {type(self.dir)}')
        elif type(self.dir) == str and not os.path.isdir(os.path.dirname(self.dir)):
            raise Exception(f"Parent direcotry doesn't exist: {os.path.dirname(self.dir)}")
        
    def load_header(self):
        self.header = LooseAtom.load_from_disk(self.header_path)

    def save_header(self):
        self.refresh_header_lately_update_time()
        self.header.save_to_disk(self.header_path)

    def clear(self):
        utils.remove_all_files_in_dir(self.dir)

    def save_to_disk(self, new_dir=None):
        if new_dir is not None:
            self.dir = new_dir
        self._check_dir()
        if os.path.isdir(self.dir):
            self.clear()
        else:
            os.mkdir(self.dir)  
        self.save_header()

    @classmethod
    def load_from_disk(cls, dir_path):
        obj = cls(dir_path=dir_path)
        obj.load_header()
        return obj
    
    @property
    def _content_str(self):
        return ''


    def __str__(self):
        try:
            self._check_dir()
            suffix_str = f'({self.dir})'
        except:
            suffix_str = '(In Memory)'

        stars_str = '*'*8
        divider_str = f'{stars_str} {self.__class__.__name__} {suffix_str} {stars_str}'
        title_str = f'name={self.name} | lately_update_time={self.lately_update_time}'
        title_str = ' '*((len(divider_str)-len(title_str))//2) + title_str
        return f"\n{divider_str}\n{'-'*len(divider_str)}\n{title_str}\n{'-'*len(divider_str)}\n{self._content_str}\n{divider_str}\n"
    
# obj_dir = 'D:/documents/AcademicDocuments/customed_python_pkgs/loosely/test'
# obj = LooseNamedObj()
# print(obj)
# obj.save_to_disk(obj_dir)
# print(obj)

# obj = LooseObj.load_from_disk(obj_dir)
# print(obj)
# obj.name = 'heihei'
# print(obj)