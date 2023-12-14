import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from loosely.dataset import *


def test_LooseDataset_create():
    save_path = utils.get_example_path('LooseDataset_example.dst')

    from uuid import uuid1    

    def it(num=100):
        for i in range(num):
            yield {
                'test1': np.random.randn(50, 50),
                'test2': random.randint(0, 10)
            }

    num = 100
    # print(list(it(num)))
    # dataset = LooseDataset.create_from_iter(save_path, it(num), num)
    dataset = LooseDataset.create_from_iter(save_path, it(num), num, meta={'name': 'Test Dataset'})


def test_LooseDataset_load():
    load_path = utils.get_example_path('LooseDataset_example.dst')
    dataset = LooseDataset.load_from_disk(load_path)
    print(dataset['25892198_9a33_11ee_992f_e02be9f58ff1'])