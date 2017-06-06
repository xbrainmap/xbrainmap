#!/usr/bin/env python 

# #########################################################################
# Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

"""
Specify project specific data in this file.
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

# Specify full path to the reconstructed image tiff files directoy
# data = 4608 x 4120 x 2910, trained = 50 x 1500 x 1800  - 4X
tiff_files_location = '/projects/mousebrain/recon_rot34_crop_cc'

# Specify full path to the Ilastik trained data file.
classifier = '/projects/classifiers/v1_4xmouse_train_data.ilp'










