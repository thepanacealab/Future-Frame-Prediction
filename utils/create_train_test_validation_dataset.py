import shutil
import os, stat
import random
import re
import glob
import numpy as np
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

train_parent_directory = sorted(glob.glob('train/*/'), key=numericalSort)


source = os.listdir("sample/")
destination = "sample_test/"
#shuf = np.random.permutation(source)

num_folds = 5

a = range(int(round(len(source)/num_folds)))
b = int(round(len(source)/num_folds))
index = random.sample(source, b)

'''
for dir in range(int(round(len(shuf)/num_folds))):
    result = []
    for i in range(0, len(source)):
        index = random.sample(source, a)
        result.append(dir[index])
        #shutil.copy(result, destination)
        #shutil.rmtree()
'''   

def remove_readonly(func, path, _):
    "Clear the readonly bit and reattempt the removal"
    os.chmod(path, stat.S_IWRITE)
    func(path)

for dir in index:
    from_dir = 'sample/'
    folder_name = dir 
    from_path = os.path.join(from_dir, folder_name)
    dest_path = os.path.join(destination, folder_name)
    shutil.copytree(from_path, dest_path)
    shutil.rmtree(from_path, onerror=remove_readonly)