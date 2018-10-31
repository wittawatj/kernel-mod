# all utility functions in kgof.util are visible.
from kgof.util import *
import autograd.numpy as np


def multi_way_split(arr, sizes):
    """
    Split rows of numpy array arr into len(sizes) partitions,
    where partition i contains sizes[i] rows.
    """
    if not sizes or sum(sizes)==0 or len(sizes)==0:
        raise ValueError('sizes cannot be empty. Was {}'.format(sizes))
    if sum(sizes) != arr.shape[0]:
        raise ValueError('Total sizes do not match the size of arr. Was {}. Size of arr: {}'.format(sizes, arr.shape[0]))
    sizes = np.array(sizes)
    splits = []
    start_ind = 0
    for i in range(len(sizes)):
        si = sizes[i]
        subarr = arr[start_ind:(start_ind+si)]
        splits.append(subarr)
        start_ind = start_ind + si
    assert len(splits) == len(sizes)
    return splits


def top_lowzerohigh(values, k=None):
    """
    * values: a 1-dim numpy array
    
    Return (A, B, C) where 
        * A = indices in ascending order of values
        * B = indices in ascending order of absolute values
            of the specified values
        * C = indices in descending order of values
    """
    N = len(values)
    if k is None:
        k = N
    A = np.argsort(values)
    C = A[-1:-(k+1):-1]
    B = np.argsort(np.abs(values))
    return A[:k], B[:k], C[:k]


def download_to(url, file_path):
    """
    Download the file specified by the URL and save it to the file specified
    by the file_path. Overwrite the file if exist.
    """

    # see https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
    import urllib.request
    import shutil
    # Download the file from `url` and save it locally under `file_name`:
    with urllib.request.urlopen(url) as response, \
            open(file_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
