import os
import numpy as np
from PIL import Image
import torch
import kmod.glo as glo
import argparse
from kmod.torch_models import Generator

img_size = 64
dataname = 'lsun'
epoch = 20
num_images = 30000

gen_model_names = {
    '1232_began': 'BEGAN_{}_G.pkl'.format(epoch),
    '3212_began': 'BEGAN_{}_G.pkl'.format(epoch),
    '1232_dcgan': 'GAN_{}_G.pkl'.format(epoch),
    '3212_dcgan': 'GAN_{}_G.pkl'.format(epoch),
}


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.resize((img_size, img_size),
                         resample=Image.BILINEAR)
    return img.convert('RGB')


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
        os.makedirs(dir_out, exist_ok=True)

    dir_model = os.path.join(dir_problem, 'models')
    batch_size = 64
    z_dim = 100
    for c in gen_model_names.keys():
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
    dirnames = os.listdir(dir_data)
    dir_out = os.path.join(dir_problem, 'data')
    if not os.path.exists(dir_out):
        os.makedirs(dir_out, exist_ok=True)

    for dirname in dirnames:
        path = os.path.join(dir_data, dirname)
        print(path, dirname)
        filenames = [os.path.join(path, fn) for fn in os.listdir(path)]
        filenames = filenames[:num_images]

        label_name = dirname
        savepath_data = '{}/{}.npy'.format(dir_out, label_name)
        # data = np.array([imread(fn) for fn in filenames])
        data = np.array([np.array(pil_loader(fn)) for fn in filenames])
        np.save(savepath_data, data)


def main():
    generate_images()
    subsample_images()


if __name__ == '__main__':
    dir_problem = os.path.join(glo.shared_resource_folder(),
                               'problems', dataname)
    dir_data = os.path.join(dir_problem, 'imgs')
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default=dir_data)
    parser.add_argument('--use_cuda', help='gpu option',
                        action='store_true')
    args = parser.parse_args()
    main()
