"""
Utility functions specific for experiments. These functions are less general
than the ones in kmod.util.  
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import kmod.util as util
import numpy as np
from scipy import linalg
from sklearn.metrics.pairwise import polynomial_kernel
import sys

import kmod.glo as glo
import os
import torch
from kmod.mnist.dcgan import Generator
from kmod.mnist.dcgan import DCGAN
import kmod.mnist.dcgan as mnist_dcgan
import kmod.net as net
import kmod.gen as gen


mnist_model_names= ['dcgan', 'began', 'wgan', 'lsgan', 'gan', 'wgan_gp',
                    'vae']
shared_resource_path = glo.shared_resource_folder()


def plot_images_grid(images, func_img=None, grid_rows=4, grid_cols=4):
    """
    Plot images in a grid, starting from index 0 to the maximum size of the
    grid.

    images: stack of images images[i] is one image
    func_img: function to run on each image before plotting
    """
    gs1 = gridspec.GridSpec(grid_rows, grid_cols)
    gs1.update(wspace=0.05, hspace=0.05)  # set the spacing between axes. 

    for i in range(grid_rows*grid_cols):
        if func_img is not None:
            img = func_img(images[i])
        else:
            img = images[i]
        
#         plt.subplot(grid_rows, grid_cols, i+1)
        plt.subplot(gs1[i])
        plt.imshow(img)
        plt.axis('off')


########################
#based on https://github.com/mbinkowski/MMD-GAN
###############################
def get_splits(n, splits=10, split_method='openai'):
    if split_method == 'openai':
        return [slice(i * n // splits, (i + 1) * n // splits)
                for i in range(splits)]
    elif split_method == 'bootstrap':
        return [np.random.choice(n, n) for _ in range(splits)]
    elif 'copy':
        return [np.arange(n) for _ in range(splits)]
    else:
        raise ValueError("bad split_method {}".format(split_method))


def inception_score(preds, **split_args):
    split_inds = get_splits(preds.shape[0], **split_args)
    scores = np.zeros(len(split_inds))
    for i, inds in enumerate(split_inds):
        part = preds[inds]
        kl = part * (np.log(part) - np.log(np.mean(part, 0, keepdims=True)))
        kl = np.mean(np.sum(kl, 1))
        scores[i] = np.exp(kl)
    return scores


def fid_permutation_test(X, Y, Z, alpha=0.01, n_permute=400, seed=893):
    assert X.shape == Y.shape
    assert X.shape == Z.shape
    XYZ = np.vstack([X, Y, Z])
    nxyz = XYZ.shape[0]
    nx = ny = X.shape[0]
    splits = 1
    split_size = X.shape[0]
    split_method = 'copy'
    split_args = {'splits': splits, 'n': split_size, 'split_method': split_method}

    with util.ContextTimer(seed) as t:
        stat = np.mean(fid_score(X, Z, **split_args)) - np.mean(fid_score(Y, Z, **split_args))
        list_fid = np.zeros((n_permute))
        with util.NumpySeedContext(seed):
            for r in range(n_permute):
                ind = np.random.choice(nxyz, nxyz, replace=False)
                indx = ind[:nx]
                indy = ind[nx:nx+ny]
                indz = ind[nx+ny:]
                codes_p = XYZ[indx]
                codes_q = XYZ[indy]
                codes_r = XYZ[indz]
                fid_xz = np.mean(fid_score(codes_p, codes_r, **split_args))
                fid_yz = np.mean(fid_score(codes_q, codes_r, **split_args))
                list_fid[r] = fid_xz - fid_yz
    pvalue = np.mean(list_fid > stat)
    results = {'alpha': alpha, 'pvalue': pvalue, 'test_stat': stat,
               'h0_rejected': pvalue < alpha, 'n_permute': n_permute,
               'time_secs': t.secs,
               }
    return results


def fid_score(codes_g, codes_r, eps=1e-6, output=sys.stdout, **split_args):
    splits_g = get_splits(**split_args)
    splits_r = get_splits(**split_args)
    assert len(splits_g) == len(splits_r)
    d = codes_g.shape[1]
    assert codes_r.shape[1] == d

    scores = np.zeros(len(splits_g))
    for i, (w_g, w_r) in enumerate(zip(splits_g, splits_r)):
        part_g = codes_g[w_g]
        part_r = codes_r[w_r]

        mn_g = part_g.mean(axis=0)
        mn_r = part_r.mean(axis=0)

        cov_g = np.cov(part_g, rowvar=False)
        cov_r = np.cov(part_r, rowvar=False)

        covmean, _ = linalg.sqrtm(cov_g.dot(cov_r), disp=False)
        if not np.isfinite(covmean).all():
            cov_g[range(d), range(d)] += eps
            cov_r[range(d), range(d)] += eps
            covmean = linalg.sqrtm(cov_g.dot(cov_r))

        scores[i] = np.sum((mn_g - mn_r) ** 2) + (
            np.trace(cov_g) + np.trace(cov_r) - 2 * np.trace(covmean))
    return scores


def polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1,
                   var_at_m=None, ret_var=True):
    # use  k(x, y) = (gamma <x, y> + coef0)^degree
    # default gamma is 1 / dim
    X = codes_g
    Y = codes_r

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return _mmd2_and_variance(K_XX, K_XY, K_YY,
                              var_at_m=var_at_m, ret_var=ret_var)


def _sqn(arr):
    flat = np.ravel(arr)
    return flat.dot(flat)


def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False,
                       mmd_est='unbiased', block_size=1024,
                       var_at_m=None, ret_var=True):
    # based on
    # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # but changed to not compute the full kernel matrix at once
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)
    if var_at_m is None:
        var_at_m = m

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2 * K_XY_sum / (m * m))
    else:
        assert mmd_est in {'unbiased', 'u-statistic'}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))

    if not ret_var:
        return mmd2

    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    K_XY_2_sum = _sqn(K_XY)

    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

    m1 = m - 1
    m2 = m - 2
    zeta1_est = (
        1 / (m * m1 * m2) * (
            _sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 1 / (m * m * m1) * (
            _sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
        - 2 / m**4 * K_XY_sum**2
        - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 2 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    zeta2_est = (
        1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 2 / (m * m) * K_XY_2_sum
        - 2 / m**4 * K_XY_sum**2
        - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 4 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
               + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)

    return mmd2, var_est


def polynomial_mmd_averages(codes_g, codes_r, n_subsets=50, subset_size=1000, ret_var=True, output=sys.stdout, **kernel_args):
    m = min(codes_g.shape[0], codes_r.shape[0])
    mmds = np.zeros(n_subsets)
    if ret_var:
        vars = np.zeros(n_subsets)
    choice = np.random.choice

    for i in range(n_subsets):
        g = codes_g[choice(len(codes_g), subset_size, replace=False)]
        r = codes_r[choice(len(codes_r), subset_size, replace=False)]
        o = polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
        if ret_var:
            mmds[i], vars[i] = o
        else:
            mmds[i] = o
    return (mmds, vars) if ret_var else mmds


def load_mnist_gen(model_name, epoch, tensor_type, batch_size=64, **load_options):
    name = model_name.lower()
    if name not in mnist_model_names:
        raise ValueError('Model name has be one of '
                          '{} and was'.format(mnist_model_names, name))
    print('Loading ', name)
    if name == 'dcgan':
        # load a model from the shared folder
        model_folder = glo.shared_resource_folder('prob_models', 'mnist_dcgan')
        model_fname = 'mnist_dcgan_ep{}_bs{}.pt'.format(epoch, batch_size)
        model_fpath = os.path.join(model_folder, model_fname)
        print('Shared resource path at: {}'.format(shared_resource_path))
        print('Model folder: {}'.format(model_folder))
        print('Model file: ', model_fname)
        # load the generator of type kmod.gen.PTNoiseTransformer
        dcgan = net.SerializableModule.load(model_fpath, **load_options)
        # make sure to respect the specified tensor_type
        dcgan.tensor_type = tensor_type
        return dcgan
    
    elif ('gan' in name):
        # load a model from the shared folder
        model_folder = glo.shared_resource_folder('prob_models', 'mnist_{}'.format(name), str(epoch))
        model_fname = '{}_G.pkl'.format(name.upper())
        model_fpath = os.path.join(model_folder, model_fname)
        print('Shared resource path at: {}'.format(shared_resource_path))
        print('Model folder: {}'.format(model_folder))
        print('Model file: ', model_fname)
        
        from kmod.mnist.began import Generator as Generator_
        # load the generator of type kmod.gen.PTNoiseTransformer
        image_size = 28
        z_dim = 62 #dimention of noise, this is fixed. so don't change
        g = Generator_(input_dim=z_dim,input_size=image_size)
        in_out_shapes = (z_dim, image_size)
        def f_sample_noise(n):
            return torch.rand((n, z_dim))
        g.load(model_fpath, **load_options)
        #print(g.fc[0].weight.is_cuda)
        gan_model = gen.PTNoiseTransformerAdapter(module=g,
                f_sample_noise=f_sample_noise, in_out_shapes=in_out_shapes,
                tensor_type=tensor_type)
        return gan_model

    elif name == 'vae':
         # load a model from the shared folder
        model_folder = glo.shared_resource_folder('prob_models', 'mnist_{}'.format(name), str(epoch))
        model_fname = '{}.pkl'.format(name.upper())
        model_fpath = os.path.join(model_folder, model_fname)
        print('Shared resource path at: {}'.format(shared_resource_path))
        print('Model folder: {}'.format(model_folder))
        print('Model file: ', model_fname)
        
        from kmod.mnist.vae import VAE
        # load the generator of type kmod.gen.PTNoiseTransformer
        image_size = 28
        z_dim = 20 #dimention of noise, this is fixed. so don't change
        g = VAE()
        in_out_shapes = (z_dim, image_size)
        def f_sample_noise(n):
            return torch.randn((n, z_dim))
        g.load(model_fpath, **load_options)
        #print(g.fc[0].weight.is_cuda)
        vae = gen.PTNoiseTransformerAdapter(module=g,
                f_sample_noise=f_sample_noise, in_out_shapes=in_out_shapes,
                tensor_type=tensor_type)
        return vae


