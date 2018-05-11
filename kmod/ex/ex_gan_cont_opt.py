# coding: utf-8
import kmod
# submodules
from kmod import data, util
from kmod import gan_ume_opt as go
from kmod import torch_models as tm

import torch
import time
import os
import pickle
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image

import pretrainedmodels
import argparse
import logging


def open_images(paths, size=64, resize=False):
    img_data = []
    for path in paths:
        im = Image.open(path)
        if resize:
            im = im.resize((size, size))
        im = np.array(im)
        img_data.append(im)
    return np.array(img_data)


def sample_images(gen, num_sample):
    batch_size = 256
    z_dim = gen.z_size
    r = range(num_sample)
    samples = []
    for i in range(0, num_sample, batch_size):
        size = len(r[i: i+batch_size])
        sample_z_ = Variable((torch.rand((size, z_dim))).view(-1, z_dim, 1, 1))
        sample_z_ = sample_z_.cuda(go.gpu_id)
        sample_z_ = -2. * sample_z_ + 1.
        sample_z_ = sample_z_.float()
        sample = gen(sample_z_).cpu().data.numpy()
        samples.append(sample)
    return np.vstack(samples)


def normalize(images, mean, std):
    """normalize ndarray images of shape N x H x W x C"""
    return (images - mean) / std


def gpu_setting(args):
    # torch.backends.cudnn.enabled = True
    gpu_id = args.gpu_id
    gpu_mode = args.gpu
    batch_size = args.batch_size
    go.set_gpu_mode(gpu_mode)
    go.set_gpu_id(gpu_id)
    go.set_batch_size(batch_size)


def run_optimization(args, gp, gq, img_data, model_name, J=10):
    """
    Wrapper for noise space optimization

    """

    model = load_pretrained_model(model_name)
    model.eval()
    if model_name == 'inceptionv3':
        feat_func = model.pool3
    else:
        feat_func = model.features

    sample_size = 1000  # number of images we want to generate
    samples_p = sample_images(gp, sample_size)
    datap = go.extract_feats(
        samples_p,
        feat_func,
        upsample=True
    )

    samples_q = sample_images(gq, sample_size)
    dataq = go.extract_feats(
        samples_q,
        feat_func,
        upsample=True
    )

    datar = util.subsample_rows(img_data, sample_size)
    datar = samples_p = go.extract_feats(
        datar.transpose((0, 3, 1, 2)),
        feat_func,
        upsample=True
    )
    datap = data.Data(datap)
    dataq = data.Data(dataq)
    datar = data.Data(datar)

    Zp0 = np.random.uniform(-1, 1, (J, gp.z_size))
    Zq0 = np.random.uniform(-1, 1, (J, gq.z_size))
    XYZ = np.vstack((datap.data(), dataq.data(), datar.data()))
    med2 = util.meddistance(XYZ, subsample=1000)**2

    with util.ContextTimer() as t:
        Z_opt, gw_opt, opt_result = go.optimize_3sample_criterion(datap, dataq,
                                                                  datar, gp, gq,
                                                                  feat_func,
                                                                  Zp0, Zq0,
                                                                  gwidth0=med2)

    results = {}
    results['Z'] = Z_opt
    results['width'] = gw_opt
    results['opt'] = opt_result
    results['t'] = t

    return results


def load_pretrained_model(model_name):
    # loading a feature extractor
    if model_name == 'inceptionv3':
        model = tm.load_inception_v3(pretrained='imagenet', gpu_id=go.gpu_id)
        go.set_model_input_size(299)
        go.set_batch_size(64)
    else:
        model = pretrainedmodels.__dict__[model_name]().cuda(go.gpu_id)
    return model


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gpu', action='store_true', default=True)
    group.add_argument('--cpu', action='store_false')
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./returned_locations')
    args = parser.parse_args()

    model_dir = '/nfs/nhome/live/heishirok/Work/kmod/problems/celeba/models/'
    gp = tm.Generator().cuda(go.gpu_id)
    gp.eval()
    gq = tm.Generator().cuda(go.gpu_id)
    gq.eval()
    gp.load('{}/GAN_G_smile_unif.pkl'.format(model_dir))
    gq.load('{}/GAN_G_nosmile_unif.pkl'.format(model_dir))

    data_dir = '/nfs/nhome/live/heishirok/Work/kmod/problems/celeba/img_align_celeba'
    test_img_list = []
    filelist_path = '/nfs/nhome/live/heishirok/Work/kmod/problems/celeba/test_list.txt'
    with open(filelist_path) as f:
        for line in f:
            test_img_list.append(line.rstrip('\n'))

    img_data = []
    for filename in test_img_list:
        data_path = '{}/{}'.format(data_dir, filename)
        im = np.array(Image.open(data_path))
        img_data.append(im)
    img_data = np.array(img_data) / 255.  # maybe better to normalize differently for tests

    model_name = 'inceptionv3'
    results = run_optimization(args, gp, gq, img_data, model_name, J=10)

    dir_path = '{}/{}'.format(args.save_dir, model_name)
    os.makedirs(dir_path, exist_ok=True)
    filename = time.strftime('%Y-%m-%d_%H-%M-%S_')
    f = open('{}/{}.pkl'.format(dir_path, filename), 'wb')
    pickle.dump(results, f)
    f.close()


if __name__ == '__main__':
    main()
