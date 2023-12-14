import os, dill, re, json
import utils, numpy as np
from collections.abc import Iterable



def get_class_by_path(path):
    assert is_processor_path(path)
    ext = utils.get_ext(path).replace('.', '')
    cls_map = dict()
    for c in SUPPORT_CLASSES:
        cls_map[c.EXT] = c
    return cls_map[ext]
    
def get_obj_by_path(path):
    return get_class_by_path(path).load(path)

class Processor:
    EXT = 'processor'

    def __init__(self, func, name=None):
        self.process = func
        self.name = (name or func.__name__).replace('<', '(').replace('>', ')')

    @classmethod
    def append_cls_ext(cls, path):
        return path + f'.{cls.EXT}'
    
    @staticmethod
    def get_name_from_path(path):
        return '.'.join(os.path.basename(path).split('.')[:-1])

    @property
    def file_name(self):
        return '.'.join([self.name, self.EXT])
    
    def get_save_path(self, save_dir):
        return os.path.join(save_dir, self.file_name)
    
    def save(self, save_dir, new_name=None):
        if new_name is not None:
            self.name = new_name
        with open(self.get_save_path(save_dir), 'wb') as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            obj = dill.load(f)
            obj.name = cls.get_name_from_path(path)
            return cls(obj.process, obj.name)
        
    def __str__(self):
        return f"<{type(self).__name__} name='{self.name}' func='{self.process.__name__}'>"

class Wrapper(Processor):
    EXT = 'wrapper'


class Filter(Processor):
    EXT = 'filter'

SUPPORT_CLASSES = [Processor, Wrapper, Filter]
def is_processor_path(path):
    ext = utils.get_ext(path).replace('.', '')
    return ext in [c.EXT for c in SUPPORT_CLASSES]

class Pipeline:
    INFO_FILE_NAME = 'info.json'

    def __init__(self, pipeline_dir=None, info=None, extra_processors=[]):
        self.dir = pipeline_dir
        self.info = self.get_default_info(info)
        self.reload_processors()
        for processor in extra_processors:
            self.append(processor)

    @property
    def info_path(self):
        return os.path.join(self.dir, self.INFO_FILE_NAME) if self.dir else None

    def reload_info(self):
        self.info = utils.load_json(self.info_path)

    def reload_processors(self):
        self.processors = []
        for file_name in self.info['order']:
            processor_path = os.path.join(self.dir, file_name)
            self.processors.append(get_obj_by_path(processor_path))

    def reload(self):
        self.reload_info()
        self.reload_processors()

    def save_info(self):
        utils.save_json(self.info_path, self.info)

    def update_info(self, new_info, partial=True, auto_save=True):
        if partial:
            self.info.update(new_info)
        else:
            self.info = new_info
        if auto_save and self.info_path:
            self.save_info()
    
    def get_default_info(self, info=None):
        init_info = {
            'order': []
        }
        if info is not None:
            init_info.update(info)
        return init_info
    
    @staticmethod
    def _check_processor(processor):
        if type(processor) == str:
            assert os.path.isfile(processor) and is_processor_path(processor)
            obj = get_obj_by_path(processor)                    
        elif isinstance(processor, Processor):
            obj = processor
        else:
            raise Exception(f'Unsupported type: {type(processor)}')
        return obj


    def insert(self, index, processor):
        processor = self._check_processor(processor)
        self.processors.insert(index, processor)
        self.info['order'].insert(index, processor.file_name)

    def append(self, processor):
        self.insert(len(self.processors), processor)

    def pop(self, index=-1):
        processor = self.processors.pop(index)
        self.info['order'].pop(index)
        return processor

    def remove(self, item, remove_all=True):
        removed_items = []
        if type(item) == str:
            indexes = [i for i, processor in enumerate(self.processors) if processor.file_name == item]
        elif isinstance(item, Processor):
            indexes = [i for i, processor in enumerate(self.processors) if processor.file_name == item.file_name]
        else:
            raise Exception(f'Unknow remove item type: {type(item)}')
        
        indexes.reverse()

        if not remove_all and len(indexes) > 1:
            indexes = indexes[:1]
        
        for index in indexes:
            removed_items.append(self.pop(index))

        return removed_items
    
    def refresh_order(self):
        self.info['order'] = [processor.file_name for processor in self.processors]

    def save_processors(self):
        for processor in self.processors:
            if not os.path.isfile(processor.get_save_path(self.dir)):
                processor.save(self.dir)

    def save(self, save_dir=None):
        assert self.dir is not None or save_dir is not None
        if save_dir is not None:
            self.dir = save_dir
        self.refresh_order()
        utils.remove_all_files_in_dir(self.dir)
        self.save_info()
        self.save_processors()
        print(f'New Pipeline saved: {self.dir}')

    @classmethod
    def load_from_dir(cls, load_dir):
        pipeline = cls(pipeline_dir=load_dir)
        pipeline.reload()
        return pipeline

    @classmethod
    def create_example(cls, pipeline_dir):
        def process1(x):
            return x+1
        processor1 = Wrapper(process1, 'plus_1')
        
        def filter2(x):
            return x > 12
        processor2 = Filter(filter2, 'gt12')
        
        processor3 = Wrapper(lambda x: x/5, 'divide5')
        pipeline = cls(pipeline_dir=pipeline_dir, extra_processors=[processor1, processor2, processor3, processor2])
        pipeline.save()
        return pipeline
        
    def __str__(self):
        content = json.dumps({
            'info_path': self.info_path,
            'info': self.info,
            'processors': [str(processor) for processor in self.processors]
        },indent=2)
        divider = '*'*8 + f' Pipeline ({self.dir if self.dir is not None else "In Memory"}) ' + '*'*8
        return f'{divider}\n{content}\n{divider}'
        
    def process_one_item(self, item, index, it):
        result = {
            'value': item,
            'passed': True
        }
        for processor in self.processors:
            param_num = utils.get_func_param_num(processor.process)
            params = [result['value'], index, it][:param_num]
            if isinstance(processor, Wrapper):
                result['value'] = processor.process(*params)
            elif isinstance(processor, Filter):
                if not processor.process(*params):
                    result['passed'] = False
                    break
            else:
                raise Exception(f'Unknow processor type during pipeline processing: {type(processor)}')
        return result
    
    def process(self, it, yield_with_index=False):
        assert isinstance(it, Iterable)
        def new_iter():
            index = 0
            for item in it:
                result = self.process_one_item(item, index, it)
                if result['passed']:
                    yield result['value'] if not yield_with_index else result['value'], index
                index += 1
        
        return new_iter()

# processor_dir = 'D:/documents/AcademicDocuments/customed_python_pkgs/neuro-dataset/example_dataset.nids'
# a = lambda x: x
# proccessor = Wrapper(a, 'wrapper1')
# proccessor.save(processor_dir)
# def c(x):
#     return x

# processor = Wrapper(a)
# print(processor.name)


# processor.save(pipeline_dir, f'1.{processor.name}')


# load_path = os.path.join(pipeline_dir, f'1.(lambda).wrapper')
# print(get_class_by_path(load_path))

# pipeline = Pipeline()
# pipeline_dir = 'D:/documents/AcademicDocuments/customed_python_pkgs/neuro-dataset/example_dataset.nids/pipeline'
# processor_path = 'D:/documents/AcademicDocuments/customed_python_pkgs/neuro-dataset/example_dataset.nids/wrapper1.wrapper'
# pipeline = Pipeline(pipeline_dir)
# print(pipeline)
# # pipeline.reload_info()
# pipeline = Pipeline(extra_processors=[processor_path, processor_path, processor_path])
# pipeline.save(pipeline_dir)

# pipeline.pop(1)
# pipeline.remove('wrapper1.wrapper')
# print(pipeline)
# pipeline.save()
# pipeline.append(processor_path)
# print(pipeline)
# pipeline.save()
# # pipeline.append(processor_path)
# # print(pipeline)
# pipeline.save(pipeline_dir)
# print(pipeline)

# pipeline = Pipeline.create_example(pipeline_dir)
# print(pipeline)

# pipeline = Pipeline.load_from_dir(pipeline_dir)
# print(pipeline)
# example_arr = [2, 12, 100, 60, 10]
# print(list(pipeline.process(example_arr, yield_with_index=True)))

# # print(Processor.get_name_from_path(load_path))
# processor2 = Processor.load(load_path)
# print(isinstance(processor2, Wrapper), issubclass(type(processor2), Filter))
# print(processor2.process(123))

# Pipeline.load_from_dir(pipeline_dir)