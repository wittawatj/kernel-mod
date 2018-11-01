# CelebA dataset

## Prepare the data
Place the celebA dataset (a directory containing jpg files) in your desired place. 
You can download from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). 
If the name of your data directory is `the_data_directory`, you should have image files as `the_data_directory/000001.jpg`. 


## Download scripts for preprocessing
From [here](http://ftp.tuebingen.mpg.de/pub/is/wittawat/kmod_share/problems/celeba/), download the following files:
1. `test_list.txt`
2. `test_smile.txt`

Place `test_list.txt` and `test_smile.txt` in `shared_resources_folder/problems/celeba`. You can change `shared_resouces_folder` in `kmod/config.py`. 

From [here](http://ftp.tuebingen.mpg.de/pub/is/wittawat/kmod_share/problems/celeba/models), 
download two trained generated models: 
1. `GAN_G_smile_unif.pkl`
2. `GAN_G_nosmile_unif.pkl`

Move them into `shared_resources_folder/problems/celeba/models`. 

## Create datasets
Use the following files in `kmod/preprocessing/celeba/` below. 
1. `dataset_lsun.py`
2. `feature_lsun.py`


In your working directory, execute the commands as follows.
1. `python dataset_celeba.py --datadir='/the/path/to/the/celebA/data/directory'`, 
2. `python feature_celeba.py`. 

You can use gpu by adding the option `--use_cuda` for both scripts. 
By default, the numpy binary files for the raw data and the inception features are created in 
`shared_resources_folder/problems/celeba`. Under this directory, you will have two sub directories `data` and `inception_features` and npy files such as `inception_features/ref_smile.npy`.
