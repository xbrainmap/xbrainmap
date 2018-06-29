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
Divides the volume into sub-volumes, cells within sub-volumes are detected and written into the data set of a HDF5 file.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import h5py
from mpi4py import MPI
import os.path
from glob import glob
from segmentation_param import *
from segment_vessels import segment_vessels

__author__ = "Mehdi Tondravi"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['compute_sub_volumes',
           'vessel_detect_big_data_mpi']

def compute_sub_volumes(cell_prob_dataset):
    """
    This function divides a given volume into sub-volumes and returns three lists. Each list has the start
    and end indices for a sub-volume. The sub-volume size is specified in the segmentation_param.py file. 
    
    Parameters
    ----------
    cell_prob_dataset : data set object 
    
    Returns
    -------
    x,y,z list of indices
    
    """
    x_sub_volumes_idx = []
    y_sub_volumes_idx = []
    z_sub_volumes_idx = []
    for x_idx in range(int(cell_prob_dataset.shape[0]/v_sub_vol_x)):
        for y_idx in range(int(cell_prob_dataset.shape[1]/v_sub_vol_y)):
            for z_idx in range(int(cell_prob_dataset.shape[2]/v_sub_vol_z)):
                x_next_idx = []
                x_next_idx.append(((v_sub_vol_x * x_idx), (v_sub_vol_x + v_sub_vol_x * x_idx)))
                x_sub_volumes_idx.append(x_next_idx)
                
                y_next_idx = []
                y_next_idx.append(((v_sub_vol_y * y_idx), (v_sub_vol_y + v_sub_vol_y * y_idx)))
                y_sub_volumes_idx.append(y_next_idx)
                
                z_next_idx = []
                z_next_idx.append(((v_sub_vol_z * z_idx), (v_sub_vol_z + v_sub_vol_z * z_idx)))
                z_sub_volumes_idx.append(z_next_idx)
    return x_sub_volumes_idx, y_sub_volumes_idx, z_sub_volumes_idx

def vessel_detect_big_data_mpi():
    """ 
    Volume is divided into several sub-volumes and a different rank detects vessel within sub-volumes. 
    Detected vessel map is written into the data set of a HDF5 file by each rank.
    
    Parameters
    ----------                                                                                             
    Input: The volume cell probability map file location is specified in the segmentation_param.py file. 
    
    Returns
    ------
    Vessel maps are written into a new data set created within the input file.                         
    """
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    name = MPI.Get_processor_name()
    
    # There is only one file ending with 'volume_prob_map.h5' in 'volume_map_file_location'.
    vol_map_file = sorted(glob(volume_map_file_location+'/*volume_prob_map.h5'))
    if not vol_map_file:
        print("*** Did not find any file ending with 'volume_prob_map.h5' ***")
        return
    
    hdf_file = h5py.File(vol_map_file[0], 'r+', driver='mpio', comm=comm)
    if not hdf_file.get('volume_vessel_probability_map', getclass=True):
        print("**** 'volume_vessel_probability_map' *** data set does not exist")
        return
    
    vessel_prob_dataset = hdf_file['volume_vessel_probability_map']
    print("vessel_prob_dataset.shape", vessel_prob_dataset.shape)
    
    # Delete the "Volume Vessel Map" data set if exists
    if hdf_file.get('Volume Vessel Map', getclass=True):
        hdf_file.__delitem__('Volume Vessel Map')
    
    # Create Data Set for the whole volume vessel map
    vol_vessel_map = hdf_file.create_dataset("Volume Vessel Map", np.shape(vessel_prob_dataset), dtype='uint8')
    
    # Divide the volume into sub-volumes. Compute start and end indices for each sub-volume
    x_sub_volumes_idx, y_sub_volumes_idx, z_sub_volumes_idx = compute_sub_volumes(vessel_prob_dataset)
    print("Done with computing sub-volumes - This is rank %d of %d running on %s" % (rank, size, name))
    
    iterations = int(len(x_sub_volumes_idx) / size)
    partial_iterations = int(len(x_sub_volumes_idx) % size)
    for idx in range(iterations):
        x_idx = x_sub_volumes_idx[rank + (size * idx)]
        y_idx = y_sub_volumes_idx[rank + (size * idx)]
        z_idx = z_sub_volumes_idx[rank + (size * idx)]
        
        vessel_prob_map = vessel_prob_dataset[x_idx[0][0] : x_idx[0][1], y_idx[0][0] : y_idx[0][1], 
                                          z_idx[0][0] : z_idx[0][1]]
        print("***Cell Sub-volume*** to be processed by rank %d x, y, z  %d:%d, %d:%d, %d:%d" % 
              (rank, x_idx[0][0], x_idx[0][1], y_idx[0][0], y_idx[0][1], z_idx[0][0], z_idx[0][1]))
        
        vessel_map = segment_vessels(vessel_prob_map, vessel_probability_threshold,
                                     vessel_dilation_size, minimum_size)
        # Below line needs more work - it will not work if vessel_map is > 2GB
        vol_vessel_map[x_idx[0][0] : x_idx[0][1], y_idx[0][0] : y_idx[0][1],
                     z_idx[0][0] : z_idx[0][1]] = vessel_map
    # Ignore the partial iteration for now - i.e. assumes the number of sub-volumes is multiple of size (process).
    print("Number of sub-volumes ignored is %d" % partial_iterations)
    hdf_file.close()

if __name__ == '__main__':
    vessel_detect_big_data_mpi()
