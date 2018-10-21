"""
Utility functions for Mnist dataset.
"""

import numpy as np
import torch 
import kmod.log as log
import kmod.plot as plot


def pt_sample_by_labels(data, label_counts):
    """
    data: a dataset such that data[i][0] is a point, and data[i][1] is an
        integer label.
    label_counts: a list of tuples of two values (A, B), where A is a label,
        and B is the count.
    """
    list_selected = []
    labels = np.array([data[i][1] for i in range(len(data))])
    for label, count in label_counts:
        inds = np.where(labels==label)[0]
        homo_data = [data[i][0] for i in inds[:count]]
        list_selected.extend(homo_data)
    # stack all
    selected = torch.stack(list_selected)
    return selected

def show_sorted_digits(imgs, digit_mapper, n_per_row=10, figsize=(8,8),
        digits=[], n_max_sample=None, normalize=True):
    """
    Show sorted generated Mnist digits.

    imgs: a Pytorch tensor of images containing a mix of digits.
    sampler: an object which implements sample(n) whose call returns a stack of
        generated MNIST digits.
    digit_mapper: a callable object which takes a stack of images and returns a
        list of digit identities as integers. This is likely a classifier.
    digits: digits (a list) to show. If not set, show all digits of 0-9.
    n_per_row: number of generated digits to show per row
    n_max_sample: total number of samples to generate from sampler.
    """
    if n_max_sample is None:
        n_max_sample = max(500, n_per_row*10*len(digits))

    Y = digit_mapper(imgs)
    UY = torch.unique(Y)

    list_row_imgs = []
    for y in UY:
        Iy = torch.nonzero(Y==y).view(-1)
        # print(Iy)
        len_Iy = len(Iy)
        if len_Iy < n_per_row:
            # not enough geneated images for the digit y
            raise ValueError('Only {} images available for digit {}. But you want to show n_per_row = {} images.'.format(len_Iy, y, n_per_row))
        imgs_y = imgs[Iy[:n_per_row]]
        list_row_imgs.append(imgs_y)

    stack_imgs = torch.cat(list_row_imgs, dim=0)
    # show the images
    plot.show_torch_imgs(stack_imgs, nrow=n_per_row, figsize=figsize,
            normalize=normalize)






