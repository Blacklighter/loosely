
from abc import ABC, abstractclassmethod, abstractproperty, abstractmethod
from enum import Enum
from pathlib import Path
import json, shutil, os
from loguru import logger
from loosely.utils import generate_id, remove_by_path

class FileType(Enum):
    DIRECTORY = 1
    FILE_JSON = 2
    FILE_NUMPY = 3
    FILE_OBJECT = 4
    FILE_FUNCTION = 5
    OTHERS = 0
    NONE = -1


def get_file_type_from_path(path):
    path = Path(path)

    def infer_by_suffix(suffix):
        if suffix == '.json':
            return FileType.FILE_JSON
        elif suffix == '.npy':
            return FileType.FILE_NUMPY
        else:
            return suffix

    if path.exists():
        if path.is_file():
            t = infer_by_suffix(path.suffix)
        else:
            t = FileType.DIRECTORY
    else:
        t = infer_by_suffix(path.suffix)
        if t == '':
            t = FileType.DIRECTORY
        logger.warning(f"{path} doesn't exist, can only be inferenced by suffix string: {t if isinstance(t, FileType) else FileType.OTHERS}")
    return t if isinstance(t, FileType) else FileType.OTHERS

def from_check_function(check_func):
    if isinstance(check_func, classmethod):
        check_func_name = check_func.__func__.__name__
    def my_decorator(func):
        def wrapper(cls_or_self, *args, **kwargs):
            try:
                check_func = getattr(cls_or_self, check_func_name)
                check_func(*args, **kwargs)
                return True
            except:
                return False
        return wrapper
    return my_decorator

class LooseBase(ABC):
    EXT = ''
    SUPPORT_INPUT_DATA_TYPE = None
    CLS_ALIAS = ''

    def __init__(self, data=None):
        load_path = None
        if type(data) == self.__class__:
            if data.can_load:
                if data.data is None:
                    load_path = data.load_path
                else:
                    data = data.data
            else:
                data = data.data
        self._check_data(data)
        self.data = data
        self.load_path = load_path

    @classmethod
    def _check_data(cls, data):
        data_type = type(data)
        if cls.SUPPORT_INPUT_DATA_TYPE is None:
            return
        elif type(cls.SUPPORT_INPUT_DATA_TYPE) == list or type(cls.SUPPORT_INPUT_DATA_TYPE) == tuple:
            if data_type not in cls.SUPPORT_INPUT_DATA_TYPE:
                raise TypeError(f"The data type to initiate '{cls.__name__}' should be one of {[t.__name__ for t in cls.SUPPORT_INPUT_DATA_TYPE]}, not '{data_type}'.")
        elif data_type != cls.SUPPORT_INPUT_DATA_TYPE:
            raise TypeError(f"The data type to initiate '{cls.__name__}' should be '{cls.SUPPORT_INPUT_DATA_TYPE}'.")

    @classmethod
    @from_check_function(_check_data)
    def is_data_valid(cls, data):
        pass
    
    
    @classmethod
    def _check_path_basically(cls, path):
        exts = Path(cls.get_file_name_from_stem('tmp')).suffixes
        path_exts = Path(path).suffixes
        if not (len(path_exts) >= len(exts) and path_exts[-len(exts):] == exts):
            raise ValueError(f"Suffixes of target path '{''.join(path_exts)}' is not matched for '{''.join(exts)}': {path}")

    @classmethod
    @from_check_function(_check_path_basically)
    def is_path_matched(cls, path):
        pass
        
    @classmethod
    def check_path_to_load(cls, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'Cannot find the file to load: {path.resolve()}')
        elif not cls.is_path_matched(path.resolve()):
            raise TypeError(f"Target File is not supported for '{cls.__name__}': {path.resolve()}")

    @classmethod
    @from_check_function(check_path_to_load)
    def is_matched_load_path(cls, path):
        pass



    @classmethod
    def _before_load(cls, load_path):
        cls.check_path_to_load(load_path)

    @abstractclassmethod
    def _load_data(cls, path):
        pass

    @property
    def can_load(self):
        return self.is_matched_load_path(self.load_path)

    def get_data(self):
        # Return data or reloaded data, auto choose whether to release
        if self.data is None:
            self.autoload()
            data = self.data
            self.release()
        else:
            data = self.data
        return data
    
    @classmethod
    def _initiate_from_loaded_data(cls, loaded_data):
        return cls(loaded_data)

    @classmethod
    def _after_load(cls, obj, load_path):
        obj.load_path = load_path

    @classmethod
    def load_from_disk(cls, load_path):
        cls._before_load(load_path)
        data = cls._load_data(load_path)
        obj = cls._initiate_from_loaded_data(data)
        cls._after_load(obj, load_path)
        return obj

    def autoload(self, load_path=None):
        if load_path is None:
            load_path = self.load_path
        if load_path is None:
            raise ValueError('load_path and self.load_path cannot be None at the same time.')
        self._before_load(load_path)
        self.data = self._load_data(load_path)
        self._after_load(self, load_path)
        

    @classmethod
    def check_path_to_save(cls, path):
        cls._check_path_basically(path)
        if Path(path).exists():
            raise FileExistsError(path)
        
    @classmethod
    def get_file_name_from_stem(cls, stem):
        return f'{str(stem)}{cls.EXT}'

    def _handle_overwrite_before_save(self, save_path):
        if Path(save_path).is_file():
            os.remove(save_path)
        # remove_by_path(save_path)
        # pass

    def _before_save(self, save_path, allow_overwrite=True, verbose=True):
        save_path = save_path if save_path is not None else self.load_path
        try:
            self.check_path_to_save(save_path)
        except FileExistsError as err_path:
            # print(self.__class__.__name__, verbose, allow_overwrite)
            if allow_overwrite:
                if verbose:
                    logger.warning(f'Ready to overwrite: {err_path}')
                self._handle_overwrite_before_save(save_path)
            else:
                raise FileExistsError(err_path)
        
        self.data = self.get_data()
        return save_path

    @abstractmethod
    def _save_data(self, path):
        '''Save all data without any check and data memory release'''
        pass
    
    def release(self):
        tmp = self.data
        self.data = None
        del tmp

    def remove_from_disk(self, verbose=False):
        self.check_path_to_load(self.load_path)
        remove_by_path(self.load_path)
        self.release()
        if verbose:
            logger.warning(f"Local '{self.__class__.__name__}' removed: {self.load_path}")

    def _after_save(self, save_path, auto_release=True, verbose=True):
        self.load_path = save_path
        if auto_release:
            self.release()
        if verbose:
            logger.success(f'{self.__class__.__name__} save complete: {save_path}')

    def save(self, save_path=None, auto_release=True, allow_overwrite=True, verbose=True):
        save_path = self._before_save(save_path, allow_overwrite, verbose)
        self._save_data(save_path)
        self._after_save(save_path, auto_release=auto_release, verbose=verbose)
        return self.get_loosely_str(Path(save_path).stem)

    @classmethod
    def build_loosely_str(cls, name, extra_str=''):
        assert type(name) == str and type(extra_str) == str
        return f"<loosely-{cls.CLS_ALIAS} '{name}' |{extra_str}|>"
    
    @property
    def extra_str(self):
        return ''

    @classmethod
    def extract_extra_from_str(cls, extra_str):
        return extra_str

    def get_loosely_str(self, name=None):
        if name is None and self.load_path is None:
            raise ValueError('Both name and load_path are None.')
        name = Path(self.load_path).stem if name is None else name
        return self.build_loosely_str(name, self.extra_str)

    def to_dict(self):
        return {
            'class': self.__class__.__name__,
            'load_path': self.load_path,
            'data': str(self.data)
        }    

    def __json__(self):
        return self.loosely_str

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)