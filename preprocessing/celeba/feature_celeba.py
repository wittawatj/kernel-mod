from kmod import torch_models as tm
import os
import numpy as np
import torch
from torch.nn.functional import interpolate
import kmod.glo as glo
import argparse

dataname = 'celeba'


def main():
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    default_type = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    dtype = torch.float
    # load_options = {} if use_cuda else {'map_location': lambda storage, loc: storage}
    torch.set_default_dtype(dtype)
    torch.set_default_tensor_type(default_type)
    model = tm.load_inception_v3(device, pretrained='imagenet',
                                 num_classes=1000)
    model.eval()
    feat_func = model.pool3

    dir_problem = os.path.join(glo.shared_resource_folder(),
                               'problems', dataname)
    dir_out = os.path.join(dir_problem, 'inception_features')
    dir_data = os.path.join(dir_problem, 'data')
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)

    size = 299
    batch_size = 64
    fig_size = 64
    for filename in os.listdir(dir_data):
        print(filename)
        label_name = os.path.splitext(filename)[0]
        data = np.load('{}/{}'.format(dir_data, filename))
        data = (data - 128.) / 128.

        n = data.shape[0]
        feat = []
        for i in range(0, n, batch_size):
            V_ = data[i:i+batch_size].reshape([-1, 3, fig_size, fig_size])
            V_ = torch.tensor(V_, dtype=dtype, device=device)
            V_ = interpolate(V_, size=size, mode='bilinear',
                             align_corners=True)
            fX = feat_func(V_).cpu().data.numpy()
            fX = fX.reshape((fX.shape[0], -1))
            feat.append(fX)
        feat = np.vstack(feat)

        path = '{}/{}'.format(dir_out, label_name)
        print('Saving to {}.npy'.format(path))
        np.save(path, feat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', help='gpu option',
                        action='store_true')
    args = parser.parse_args()
    main()
