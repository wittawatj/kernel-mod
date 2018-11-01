from __future__ import print_function
import torch
import numpy as np
from kmod.torch_models import Generator
import kmod.glo as glo
from scipy.misc import imread
import os
import argparse

dataname = 'celeba'
gen_model_names = {'gen_smile': 'GAN_G_smile_unif.pkl',
                   'gen_nonsmile': 'GAN_G_nosmile_unif.pkl'}
num_images = 16000


def generate_images():
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    default_type = (torch.cuda.FloatTensor if use_cuda
                    else torch.FloatTensor)
    dtype = torch.float
    load_options = {} if use_cuda else {'map_location':
                                        lambda storage, loc: storage}
    torch.set_default_dtype(dtype)
    torch.set_default_tensor_type(default_type)

    dir_out = os.path.join(dir_problem, 'data')
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    dir_model = os.path.join(dir_problem, 'models')
    batch_size = 64
    z_dim = 100
    for c in gen_model_names.keys():
        if c.startswith('gen_'):
            torch.manual_seed(66)

            model_path = os.path.join(dir_model, gen_model_names[c])
            generator = Generator().to(device)
            generator.load(model_path, **load_options)
            generator.eval()

            gen_imgs = []
            for _ in range(0, num_images, batch_size):
                z = torch.rand(batch_size, z_dim)
                z = z.view(-1, z_dim, 1, 1).uniform_(-1, 1).to(device)
                samples = generator(z)
                samples = samples.cpu().data.numpy()
                gen_imgs.append(samples)
            gen_imgs = np.vstack(gen_imgs)[:num_images]
            gen_imgs = (gen_imgs * 255).astype(np.uint8)
            filepath = '{}/{}.npy'.format(dir_out, c)
            print('Saving to {}'.format(filepath))
            np.save(filepath, gen_imgs)


def subsample_images():
    dir_data = args.datadir
    dir_out = os.path.join(dir_problem, 'data')
    datalist_path = os.path.join(dir_problem, 'test_list.txt')
    test_img_list = [line.rstrip('\n')
                     for line in open(datalist_path)]
    smilelist_path = os.path.join(dir_problem, 'test_smile.txt')
    smile_img_list = [line.rstrip('\n')
                      for line in open(smilelist_path)]
    nonsmile_img_list = [fn for fn in test_img_list
                         if fn not in smile_img_list]

    smile_data = []
    for filename in smile_img_list[:num_images]:
        filepath = os.path.join(dir_data, filename)
        img = imread(filepath)
        img = img.reshape((1,) + img.shape)
        smile_data.append(img)
    smile_data = np.vstack(smile_data).transpose(0, 3, 1, 2)
    smile_path = os.path.join(dir_out, 'ref_smile.npy')
    print('Saving to {}'.format(smile_path))
    np.save(smile_path, smile_data)
    del smile_data

    nonsmile_data = []
    for filename in nonsmile_img_list[:num_images]:
        filepath = os.path.join(dir_data, filename)
        img = imread(filepath)
        img = img.reshape((1,) + img.shape)
        nonsmile_data.append(img)
    nonsmile_data = np.vstack(nonsmile_data).transpose(0, 3, 1, 2)
    nonsmile_path = os.path.join(dir_out, 'ref_nonsmile.npy')
    print('Saving to {}'.format(nonsmile_path))
    np.save(nonsmile_path, nonsmile_data)
    del nonsmile_data


def main():
    generate_images()
    subsample_images()


if __name__ == '__main__':
    dir_problem = os.path.join(glo.shared_resource_folder(),
                               'problems', dataname)
    dir_data = os.path.join(dir_problem, 'img_align_celeba')
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=dir_data)
    parser.add_argument('--use_cuda', help='gpu option',
                        action='store_true')
    args = parser.parse_args()
    main()
