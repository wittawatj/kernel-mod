
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
    'scratch_path': '/nfs/data3/wittawat/tmp/',

    # Full path to the directory to store experimental results.
    'expr_results_path': '/nfs/data3/wittawat/kmod/results/',

    # Full path to the data directory
    'data_path': os.path.join(os.path.dirname(glo.get_root()), 'data')
}

