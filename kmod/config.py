
"""
This file defines global configuration of the project.
Casual usage of the package should not need to change this. 
"""

import kmod.glo as glo
import os

# This dictionary of keys is used only when scripts under kmod/ex/ are
# executed. 
expr_configs = {
    # Full path to the directory to store temporary files when running
    # experiments.     
    'scratch_path': '/is/ei/wittawat/tmp/kmod',

    # Slurm partitions.
    # When using SlurmComputationEngine for running the experiments, the partitions (groups of computing nodes)
    # can be specified here. Set to None to not set to any value (i.e., use the default partition).
    # The value is a string. For more than one partition, set to, for instance, "wrkstn,compute".
    'slurm_partitions': None,

    # Full path to the directory to store experimental results.
    'expr_results_path': '/is/ei/wittawat/results/kmod',


    # Full path to the problems directory
    # A "problems" directory contains subdirectories, each containing all files
    # related to that particular problem e.g., cifar10, LSUN, etc.
    'problems_path': os.path.join(os.path.dirname(glo.get_root()), 'problems'),

    # full path to the problem-model directory.
    # A problem-model directory contains all model-specific files (and other
    # files such as generated images during training of GANs, for instance)
    # related to that particular problem e.g., MNIST, LSUN.
    'prob_model_path': os.path.join(os.path.dirname(glo.get_root()), 'prob_models'),
 

    # Full path to the data directory
    'data_path': os.path.join(os.path.dirname(glo.get_root()), 'data'),
}

