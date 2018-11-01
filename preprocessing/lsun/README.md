# LSUN dataset

## Prepare the data
From [here](http://ftp.tuebingen.mpg.de/pub/is/wittawat/kmod_share/problems/lsun/), download the files: 
* `bedroom.tgz`
* `restaurant.tgz`
* `kitchen.tgz`
* `confroom.tgz`

Decompress them and put the four directories in your desired place. 
For example, if the name of your data directory is `the_data_directory`, you will have 
directories as 

* `the_data_directory/bedroom`
* `the_data_directory/restaurant`
* `the_data_directory/kitchen`
* `the_data_directory/confroom`

and png files such as `the_data_directory/bedroom/000038527b455eaccd15e623f2e229ecdbceba2b.png`. 


## Download scripts for preprocessing
From [here](http://ftp.tuebingen.mpg.de/pub/is/wittawat/kmod_share/problems/lsun/), 
download trained generated models,
1. `1232_began.tgz`
2. `3212_began.tgz`
3. `1232_dcgan.tgz`
4. `3212_dcgan.tgz`

decompress and move the following files
1. `1232_began/BEGAN_20_G.pkl`
2. `3212_began/BEGAN_20_G.pkl`
3. `1232_dcgan/GAN_20_G.pkl`
4. `3212_dcgan/GAN_20_G.pkl`

into `shared_resources_folder/problems/lsun/models`. 
You can change `shared_resouces_folder` in `kmod/config.py`. 

## Create datasets
Use the following files in `kmod/preprocessing/lsun/` below. 
1. `dataset_lsun.py`
2. `feature_lsun.py`


In your working directory, execute the commands as follows.
1. `python dataset_lsun.py --datadir='/the/path/to/the/lsun/data/directory'`, 
2. `python feature_lsun.py`. 

You can use gpu by adding the option `--use_cuda` for both scripts. 
By default, the numpy binary files for the raw data and the inception features are created in 
`shared_resources_folder/problems/lsun`. Under this directory, you will have two sub directories `data` and `inception_features` and npy files such as `inception_features/restaurant.npy`.
