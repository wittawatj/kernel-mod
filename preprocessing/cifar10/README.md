# CIFAR-10 dataset

## Prepare the data
Download the python version of CIFAR-10 dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html)) and decompress in your desired directory. 
You will have a directory such as `the_desired_direcotry/cifar-10-batches-py`.


## Create datasets
Use the following files in `kmod/preprocessing/cifar10/` below. 
1. `dataset_lsun.py`
2. `feature_lsun.py`

In your working directory, execute the above scripts as follows.
1. `python dataset_cifar10.py --datadir='/the_desired_direcotry/cifar-10-batches-py'`
2. `python feature_cifar10.py`

For `feature_cifar10.py`, you can use gpu by adding the option `--use_cuda`. 
By default, the numpy binary files for the raw data and the inception features are created in 
`shared_resources_folder/problems/cifar10`. Under this directory, you will have two sub directories `data` and `inception_features` and npy files such as `inception_features/frog.npy`.
You can change `shared_resouces_folder` in `kmod/config.py`. 

