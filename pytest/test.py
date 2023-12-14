import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
# from loosely.base import LooseBase, FileType
import numpy as np
from loosely import utils, tree
from loosely.atom import LooseAtomDict, LooseAtomArray, LooseAtomFunc
import inspect
# def print_type():
#     print(FileType)

def test1():
    # path = 'D:/documents/AcademicDocuments/customed_python_pkgs/loosely/examples/test1.npy'
    # l = np.array([[1,2,3], [2], 'test'])
    # l2 = np.array([[1,2], [3,4], [5,6.1]])
    # print(l.dtype == object)
    # print(LooseAtomFunc.SUPPORT_DATA_TYPE == type(len))
    # atom = LooseAtomDict({'1':123})
    # s = [atom, atom, atom]
    # s = list(set(s))
    # print(s)
    

    for i in utils.get_keys({'test1': 1, 'test32': 2}):
        print(i)
    
    # print(inspect.is({}.keys))
    # print(isinstance(0, FileType))
    # print(l2.dtype == int)
    # l.
# print_type()

def test_LooseAtomDict_load():
    # obj = LooseAtomDict({'tes':12})
    load_path = utils.get_example_path('LooseAtomDict_example.json')
    print(load_path)
    obj = LooseAtomDict.load_from_disk(load_path)
    print(obj)
    print(obj.loosely_str)
    print(utils.extract_loosely_str(obj.loosely_str))
    

def test_LooseAtomDict_create():
    save_path = utils.get_example_path('LooseAtomDict_example.json')
    data = {'test': 111}
    obj = LooseAtomDict(data)
    print(obj)
    obj.save(save_path)
    print(obj)

def test_LooseAtomArray_load():
    load_path = utils.get_example_path('LooseAtomArray_example.npy')
    obj = LooseAtomArray.load_from_disk(load_path)
    print(obj)
    print(obj.get_loosely_str())
    print(utils.extract_loosely_str(obj.get_loosely_str()))
    

def test_LooseAtomArray_create():
    save_path = utils.get_example_path('LooseAtomArray_example.npy')
    data = [[1,2,3], 'test', [None]]
    obj = LooseAtomArray(data)
    print(obj)
    obj.save(save_path)
    print(obj)

def test_LooseAtomFunc_load():
    load_path = utils.get_example_path('LooseAtomArray_example.func')
    obj = LooseAtomFunc.load_from_disk(load_path)
    print(obj)
    print(obj.loosely_str)
    

def test_LooseAtomFunc_create():
    save_path = utils.get_example_path('LooseAtomArray_example.func')
    func = lambda x: x+1
    obj = LooseAtomFunc(func)
    print(obj)
    obj.save(save_path, verbose=True)
    print(obj)


def test_LooseDict_create():
    save_path = utils.get_example_path('LooseDict_example.dict')
    # l = np.array('test')
    # print(l, type(l), l.dtype)
    # print(tree.LooseDict.is_matched_load_path(load_path))
    d = LooseAtomDict({
            'test7': 11
        })

    a = {
        'test': 11,
        't1': np.array(123),
        'test2': '1',
        'test3': [None, 2, {
            'test3_1': 6,
            'test3_2': '11'
        }, np.array([1,2,3])],
        'test4': tree.LooseDict({
            'test5': np.array([2,2,2]),
            'test6': d,
            'test7': {
                'test7_1':1,
                'test7_2': np.array([3,3,3])
            },
            'test8': LooseAtomArray([1,1,1]),
            'test9': tree.LooseDict({
                'test9_1': 1,
                'test9_2': LooseAtomDict({})
            })
        })
    }
    d = tree.LooseDict(a)
    # d.release()
    # print(d)
    d.save(save_path)
    # # LooseAtomArray([0,0]).save('D:/documents/AcademicDocuments/customed_python_pkgs/loosely/examples/LooseDict_example.dict/test4.dict/test.npy')
    # # print(d['test4']['test8'])
    # d.save(save_path)
    # # for v, trace, parent in utils.deep_iter_items(d):
    # #     print(trace)
    # # print(d)
    # # print(d['test4']['test9'])
    # # print(d)
    # # print(d)
    # # d.release()
    # # d.save()
    # # print(d['t1'])

def test_LooseDict_load():
    load_path = utils.get_example_path('LooseDict_example.dict')
    d = tree.LooseDict.load_from_disk(load_path)
    print(d)
    # d['test11'] = np.array([1,0])
    d['test4'] = {
        'test5': np.array([2,2,2]),
        'test6': tree.LooseDict({
            'tets6_1': 111
        })
    }
    # print(d['test4'].get_data())
    # for v, trace, parent in utils.deep_iter_items(d, iter_get_data=False):
    #     print(v, '111')
    # print(d['test3'][3])
    # print(tree.LooseDict.is_matched_load_path(load_path))

def test_LooseDict_setter():
    load_path = utils.get_example_path('LooseDict_example.dict')
    d = tree.LooseDict.load_from_disk(load_path)
    print(d)
    d['test4']['test5'] = np.array([1,1,1,1])
    print(d)

def test_LooseArray_create():
    data = [
        1213,
        '123',
        {
            'test1': 1,
            'test2': np.array([1,1,1]),
            'test3': [
                None, tree.LooseDict({
                    'test3_1': 'ttt',
                    'test3_2': LooseAtomArray([1,2,3]),
                    'test3_3': tree.LooseArray([1,2,3])
                })
            ]
        },
        tree.LooseArray([
            {},
            [1,2,3],
            np.array([2,2,2])
        ])
    ]
    save_path = utils.get_example_path('LooseArray_example.arr')
    tree.LooseArray(data).save(save_path)

def test_LooseArray_load():
    load_path = utils.get_example_path('LooseArray_example.arr')
    arr = tree.LooseArray.load_from_disk(load_path)
    print(arr)
    # test_arr = [1,2,3]
    # del test_arr[1]
    # print(test_arr)
    # del arr[1]
    # print(arr[2]['test2'])
    # print(arr[-1])

def test_LooseArray_setter():
    load_path = utils.get_example_path('LooseArray_example.arr')
    arr = tree.LooseArray.load_from_disk(load_path)
    print(arr)
    # del arr[0]
    # arr.insert(2, {
    #     'test': np.array([1,0])
    # })
    arr.append({
        'test': np.array([1,0])
    })
    print(arr)

def test_get_cls_from_path():
    load_path = utils.get_example_path('LooseDict_example.dict')
    print(utils.get_loose_class_by_load_path(load_path))

def test_deep_find():
    a = {
        'test': 11,
        'test2': '1',
        'test3': [None, 2, {
            'test3-1': 6,
            'test3-2': '11'
        }]
    }

    print(utils.find_deep_index(a))
    print(utils.find_deep_index(a, lambda item: item is None))


def test_deep_map():
    a = {
        'test': 11,
        'test2': '1',
        'test3': [None, 'thah', 2, {
            'test3-1': 6,
            'test3-2': '11'
        }]
    }

    print(utils.deep_map(a, lambda item: item, lambda item: type(item)==int))
    # print(utils.find_deep_index(a, lambda item: item is None))