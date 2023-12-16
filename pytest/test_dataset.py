import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from loosely.atom import LooseAtomNifti
from loosely.dataset import *


def test_LooseDataset_create():
    save_path = 'D:/documents/AcademicDocuments/customed_python_pkgs/loosely/examples/LooseDataset_example.dst'

    from uuid import uuid1    

    def it(num=100):
        for i in range(num):
            yield {
                'test1': np.random.randn(121, 145, 121),
                'test2': random.randint(0, 10)
            }

    num = 100
    # print(list(it(num)))
    # dataset = LooseDataset.create_from_iter(save_path, it(num), num)
    import nibabel 
    template_path = 'D:/documents/AcademicDocuments/MasterCandidate/research/experiments/atlas/BNA/BN_Atlas_246_1mm.nii.gz'
    dataset = LooseDataset.create_from_iter(save_path, it(num), num, 
                                            meta={'name': 'Test Dataset'})
    dataset.extra['template'] = LooseAtomNifti(template_path)
    print(dataset.extra)
    print(dataset.extra['template'])


def test_LooseDataset_load():
    # load_path = utils.get_example_path('LooseDataset_example.dst')
    load_path = 'D:/documents/AcademicDocuments/customed_python_pkgs/loosely/examples/LooseDataset_example.dst'
    dataset = LooseDataset.load_from_disk(load_path)
    torch_dataset = dataset.to_torch_dataset()
    print(torch_dataset[1])
    # print(dataset.table[1]['test1'])
    # print(dataset.meta)
    # print(dataset.meta['extra'].get_data())
    # print(torch_dataset)
    # print(torch_dataset[1])
    