"""
Specify segmentation project data in this file.

Must specify the subvolume dimensions, TIFF file location and training
data file location in this file
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Mehdi Tondravi"
__copyright__ = "Copyright (c) 2017, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

'''
User should specify the following info:

1) The sub-volume dimensions: il_sub_vol_x (number of slices), il_sub_vol_y (columns) and il_sub_vol_z (rows)
2) tiff_files_location - the full path to the directory containing TIFF image files.
3) classifier - the full path to the directory containing the Ilastik trained data file.
'''

# Subvolume dimensions for breaking up the volume image.
# il_sub_vol_x = number of slices
# il_sub_vol_y = number of columns
# il_sub_vol_z = number of rows
il_sub_vol_x = 1536
il_sub_vol_y = 1455
il_sub_vol_z = 2060

# Specify full path to the reconstructed image tiff files directory
tiff_files_location = '/projects/mousebrain/recon_rot34_crop_cc'

# Specify full path to the Ilastik trained data file.
classifier = '/projects/classifiers/v1_4xmouse_train_data.ilp'











