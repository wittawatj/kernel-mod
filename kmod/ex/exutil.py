"""
Utility functions specific for experiments. These functions are less general
than the ones in kmod.util.  
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import kmod.util as util

def plot_images_grid(images, func_img=None, grid_rows=4, grid_cols=4):
    """
    Plot images in a grid, starting from index 0 to the maximum size of the
    grid.

    images: stack of images images[i] is one image
    func_img: function to run on each image before plotting
    """
    gs1 = gridspec.GridSpec(grid_rows, grid_cols)
    gs1.update(wspace=0.05, hspace=0.05) # set the spacing between axes. 

    for i in range(grid_rows*grid_cols):
        if func_img is not None:
            img = func_img(images[i])
        else:
            img = images[i]
        
#         plt.subplot(grid_rows, grid_cols, i+1)
        plt.subplot(gs1[i])
        plt.imshow(img)
        plt.axis('off')
