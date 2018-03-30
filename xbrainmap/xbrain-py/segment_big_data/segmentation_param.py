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
This file has various parameters values needed for cell and vessel segmentation of big data.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import multiprocessing
from psutil import virtual_memory
import socket

__author__ = "Mehdi Tondravi"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


no_of_threads = multiprocessing.cpu_count()
ram_size = int(virtual_memory().total/(1024**3)) * 1000


host_name=socket.getfqdn()

if 'alcf' in host_name:
#    print("**** host_name *****", host_name)
    tiff_files_location = '/projects/NeuroTomoDemo/tondravi/eva_block'
    classifier = '/projects/NeuroTomoDemo/tondravi/classifiers/my_s4_block_2.ilp'
#    classifier = '/projects/NeuroTomoDemo/tondravi/classifiers/ilastik_test_s4.ilp'
    
#    tiff_files_location = '/projects/NeuroTomoDemo/tondravi/eshrew080216'
#    tiff_files_location = '/projects/NeuroTomoDemo/tondravi/eshrew080216_all/recon'
#    classifier = '/projects/NeuroTomoDemo/tondravi/classifiers/MyEshrew0802.ilp'
    
# Cell and vessel detection sub-volume sizes
    sub_vol_x = 140
    sub_vol_y = 182
    sub_vol_z = 253
    
    v_sub_vol_x = 280
    v_sub_vol_y = 364
    v_sub_vol_z = 253

else:
    tiff_files_location = '/Users/mehditondravi/Downloads/eva_block'
    classifier = '/Users/mehditondravi/xray_data/train_data/my_s4_block_2.ilp'
#    classifier = '/Users/mehditondravi/playground/mat_to_python/ilastik_play/xbrain/library/ilastik_classifiers/ilastik_test_s4.ilp'
# Cell and vessel detection sub-volume sizes
    sub_vol_x = 140
    sub_vol_y = 182
    sub_vol_z = 253
    
    v_sub_vol_x = 280
    v_sub_vol_y = 364
    v_sub_vol_z = 253

hdf_files_location = tiff_files_location + '_mpi_hdf'
volume_map_file_location = tiff_files_location + '_volume_prob_maps'

cell_label_idx = 2
vessel_label_idx = 1

# Cell detection parameters                                                                         
cell_probability_threshold  = 0.2
stopping_criterion = 0.47
initial_template_size = 18
dilation_size = 8
max_no_cells = 2

# Vessel segmentation parameters                                                                    
vessel_probability_threshold = .68
vessel_dilation_size = 3
minimum_size = 4000
