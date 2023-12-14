import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from loosely import utils
import numpy as np


def test_trace_all_leaves():
    a = {
        'test': 11,
        'test2': '1',
        'test3': [np.array(123), 'thah', 2, {
            'test3-1': 6,
            'test3-2': '11'
        }]
    }

    # print(list({1,2}))
    r = list(utils.trace_all_leaves(a))
    print(list(zip(*r))[1])


def test_deep_copy():
    a = {
        'test': 11,
        'test2': '1',
        'test3': [np.array(123), 'thah', 2, {
            'test3-1': 6,
            'test3-2': '11'
        }]
    }

    # print(list({1,2}))
    r = utils.deep_copy(a)
    print(r)
    r['test3'][3]['test3-2'] = 6
    print(r)
    print(a)

def test_loosely_item_funcs():
    item_str = "<loosely-numpy 'test3,3.npy'>"
    item_str2 = "<loosely-  't'>"
    # print(utils.is_loosely_item_str(item_str))
    # print(utils.is_loosely_item_str(item_str2))
    print(utils.extract_all_info_from_item_str(item_str))
    print(utils.extract_all_info_from_item_str(item_str2))

    print(utils.get_loosely_item_str('numpy', ['test3', 3], '.npy'))