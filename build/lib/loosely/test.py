# from LooseIterable import LooseList
# from LooseDataset import LooseDataset
# from process import Wrapper

# list_dir = 'D:/documents/AcademicDocuments/customed_python_pkgs/loosely/example_dataset3.nids/table'
# loose_list = LooseList.load_from_disk(list_dir)

# print(loose_list)

# dataset_path = 'D:/documents/AcademicDocuments/customed_python_pkgs/loosely/example_dataset3.nids'
# dataset = LooseDataset.create_example(dataset_path, n=100)
# dataset = LooseDataset.load_from_disk(dataset_path)

# print(dataset.table[30]['data.zip'].get_data())
# dataset.table[24]['data.zip']['d'] = 90
    # print(len(dataset.table))

def once(a, b):
    print(a+b)
    return a+b
    
def y(a, b):
    return a-b

from unit_run import Unit
unit_dir = 'D:/documents/AcademicDocuments/customed_python_pkgs/unit-run/tests/test'
src_path = 'D:/documents/AcademicDocuments/customed_python_pkgs/unit-run/tests/test.py'
name = 'y'

unit = Unit(name, src_path)