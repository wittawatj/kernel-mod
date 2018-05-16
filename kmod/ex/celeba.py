"""
Code for running experiments on the celeba data.

By default, assume the following directory structure:


kmod/problems/cifar10/
├── data
│   ├── gen_smile.npy
│   ├── gen_nonsmile.npy
│   ├── ref_smile.npy
│   └── ref_smile.npy
├── inception_features
│   ├── gen_smile.npy
│   ├── gen_nonsmile.npy
│   ├── ref_smile.npy
│   └── ref_nonsmile.npy

path to celeba/ can be changed in kmod.config.py
"""
from kmod import util, glo, log
import autograd.numpy as np
import os

celeba_folder = glo.problems_file('celeba')
celeba_classes = ['gen_smile', 'gen_nonsmile', 'ref_smile', 'ref_nonsmile']
celeba_class_ind_dict = dict(zip((celeba_classes), range(len(celeba_classes))))


def celeba_file(*relative_path):
    return os.path.join(celeba_folder, *relative_path)


def load_data_array(class_name):
    """
    class_name can be gen_smile, ref_nonsmile, ....
    """
    npy_path = celeba_file('data', '{}.npy'.format(class_name))
    array = np.load(npy_path)
    return array


def load_feature_array(class_name, feature_folder='inception_features'):
    """
    class_name can be airplane, automobile, ... or wholedata.
    """
    npy_path = celeba_file(feature_folder, '{}.npy'.format(class_name))
    array = np.load(npy_path)
    return array


def load_stack(class_data_loader, classes=None, seed=28, max_class_size=None):
    """
    This function is deterministic given the seed.
    
    max_class_size: if specified, then randomly select at most max_class_size 
        points in each class.
    """
    if classes is None:
        classes = get_classes()
    list_arrays = []
    label_arrays = []
    with util.NumpySeedContext(seed=seed):
        for c in classes:
            log.l().info('Loading celeba class: {}'.format(c))
            arr = class_data_loader(c)
            nc = arr.shape[0]
            if max_class_size is not None:
                ncmax = min(nc, max_class_size)
            else:
                ncmax = nc
            Ind = util.subsample_ind(nc, ncmax, seed=seed+3)
            sub_arr = arr[Ind, :]
            class_label = celeba_class_ind_dict[c]
            Yc = np.ones(ncmax)*class_label

            list_arrays.append(sub_arr)
            label_arrays.append(Yc)
    stack = np.vstack(list_arrays)
    label_stack = np.hstack(label_arrays)
    assert stack.shape[0] <= len(classes)*max_class_size
    assert stack.shape[0] == len(label_stack)
    return stack, label_stack


def load_stack_data(classes=None, seed=28, max_class_size=None):
    """
    Load all the numpy array data (images) in the specified list of classes (as strings).
    If None, load all 10 classes.
    
    return (X, Y) where X = stack of all the data, Y = one-dim numpy array of class indices
    """
    return load_stack(load_data_array, classes, seed, max_class_size)


def load_stack_feature(classes=None, seed=28, max_class_size=None):
    """
    Load all the numpy array of features in the specified list of classes (as strings).
    If None, load all 10 classes.
    
    return (X, Y) where X = stack of all the features, Y = one-dim numpy array of class indices
    """
    return load_stack(load_feature_array, classes, seed, max_class_size)


def get_classes():
    return celeba_classes
