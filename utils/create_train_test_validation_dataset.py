import shutil
import os, stat
import random
import re
import glob
import numpy as np
import argparse


def is_dir(dirname):
    """Checks if a path is an actual directory"""
    if not os.path.isdir(dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        #msg = "{0} is not a directory".format(dirname)
        #raise argparse.ArgumentTypeError(msg)
    else:
        
        return os.path.abspath(os.path.realpath(os.path.expanduser(dirname)))

parser = argparse.ArgumentParser()

parser.add_argument('source', type=is_dir)
parser.add_argument('test_dest', type=is_dir)
parser.add_argument('valid_dest', type=is_dir)
args = parser.parse_args()


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

#train_parent_directory = sorted(glob.glob('train/*/'), key=numericalSort)

src_dir = args.source
#source = os.listdir(src_dir)
test_destination = args.test_dest
validation_destination = args.valid_dest
#shuf = np.random.permutation(source)

num_folds = 5

#a = range(int(round(len(source)/num_folds)))
#b = int(round(len(source)/num_folds))
#index = random.sample(source, b)


def create_test_dataset(src_dir):
    source = os.listdir(src_dir)
    b = int(round(len(source)/num_folds))
    index = random.sample(source, b)
    for dir in index:
        folder_name = dir 
        src_path = os.path.join(src_dir, folder_name)
        dest_path = os.path.join(test_destination, folder_name)
        shutil.copytree(src_path, dest_path)
        shutil.rmtree(src_path, onerror=remove_readonly)
        
        
def create_validation_dataset(src_dir):
    source = os.listdir(src_dir)
    b = int(round(len(source)/num_folds))
    index = random.sample(source, b)
    for dir in index:
        folder_name = dir 
        src_path = os.path.join(src_dir, folder_name)
        dest_path = os.path.join(validation_destination, folder_name)
        shutil.copytree(src_path, dest_path)
        shutil.rmtree(src_path, onerror=remove_readonly)

        
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

create_test_dataset(src_dir)

#update the source with the updated folder list
#source = os.listdir(src_dir)
#b = int(round(len(source)/num_folds))
#index = random.sample(source, b)
create_validation_dataset(src_dir)