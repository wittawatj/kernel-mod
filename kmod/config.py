
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
    'scratch_path': '/home/nuke/git/tmp/kmod/',

    # Slurm partitions.
    # When using SlurmComputationEngine for running the experiments, the paritions (groups of computing nodes)
    # can be specified here. Set to None to not set to any value (i.e., use the default partition).
    # The value is a string. For more than one partition, set to, for instance, "wrkstn,compute".
    'slurm_partitions': None,

    # Full path to the directory to store experimental results.
    'expr_results_path': '/home/nuke/git/results/kmod/',

    # Full path to the data directory
    'data_path': os.path.join(os.path.dirname(glo.get_root()), 'data')
}

