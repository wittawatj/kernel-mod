#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from scipy import linalg
from six.moves import cPickle as pickle

import argparse
import glob
import numpy as np
import os
import sys
import kmod.glo as glo

dataname = 'cifar10'


def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()

    return data


def preprocessing(data):
    mean = np.mean(data, axis=0)
    mdata = data - mean
    sigma = np.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = linalg.svd(sigma)
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S))), U.T)
    whiten = np.dot(mdata, components.T)

    return components, mean, whiten


def main():
    dir_out = os.path.join(dir_problem, 'data')
    if not os.path.exists(dir_out):
        os.makedirs(dir_out, exist_ok=True)

    # prepare dataset
    data = np.zeros((60000, 3 * 32 * 32), dtype=np.float)
    labels = []
    dir_data = args.datadir
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    file_paths = sorted(glob.glob('{}/data_batch*'.format(dir_data)))
    file_paths += glob.glob('{}/test_batch'.format(dir_data))
    for i, data_fn in enumerate(file_paths):
        batch = unpickle(data_fn)
        data[i * 10000:(i + 1) * 10000] = batch['data']
        labels.extend(batch['labels'])
    meta_data = unpickle('{}/batches.meta'.format(dir_data))
    label_names = meta_data['label_names']
    for label, label_name in enumerate(label_names):
        idx = (np.array(labels) == label)
        sub_data = data[idx]
        np.save('{}/{}'.format(dir_out, label_name), sub_data)
    np.save('{}/wholedata'.format(dir_out), data)


if __name__ == '__main__':
    dir_problem = os.path.join(glo.shared_resource_folder(),
                               'problems', dataname)
    dir_data = os.path.join(dir_problem, 'cifar10-10-batches_py')
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str,
                        default=dir_data)
    args = parser.parse_args()
    main()
