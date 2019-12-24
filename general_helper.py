
import sys
import os
from os.path import exists, splitext, join
from time import sleep
import pickle
import cv2

def get_file_list_recursively(root_dir, allowed_extensions= [], verbose= False):
    '''
    returns a list of full paths of all files under "root_dir"
    If a list of allowed extensions is provided, files are filtered

    parameters:
    ______________
    root_dir: string
        root of the hierarchy
    allowed_extensions: list
        list of extensions to filter

    return:
    _____________
    file_list: list
        list full path of files under root_dir
    
    '''

    if not exists(root_dir):
        raise ValueError('Directory "{}" doens\'t exist', format(root_dir))

    file_list= []
   
    for dir, sub_dirs, files in os.walk(root_dir):
        counter=0
        for file in files:
            file_name, file_ext= splitext(file)
            if allowed_extensions and file_ext not in allowed_extensions:
                pass
            else:
                counter+= 1
                file_list.append(os.path.join(dir, file))
                if verbose:
                    sys.stdout.write('\r[{}] - found {:06d} files...'.format(dir.split("\\")[-1][:20], counter))
                    sys.stdout.flush()
        if verbose: print("")
                
    return file_list


def save(something, something_name):
    filename= something_name+ ".pickle"
    with open(filename, "wb") as f:
        pickle.dump(something, f)

def load(something_path):
    filename= something_path+ ".pickle"
    with open(filename, "rb") as f:
        temp= pickle.load(f)
    
    return temp 