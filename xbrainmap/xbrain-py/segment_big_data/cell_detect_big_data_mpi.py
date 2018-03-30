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
import pdb
import numpy as np
import h5py
from mpi4py import MPI
import os.path
from glob import glob
from segmentation_param import *
from detect_cells import detect_cells

__author__ = "Mehdi Tondravi"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['compute_sub_volumes',
           'cell_detect_big_data_mpi']

def compute_sub_volumes(cell_prob_dataset):
    """
    This function divides a given volume into sub-volumes and returns three lists. Each list has the start
    and end indices for a sub-volume. The sub-volume size is specified in the segmentation_param.py file. 
    
    Parameters
    ----------
    cell_prob_dataset : data set object
    
    Returns
    ------
    x,y,z list of indices 
    
    """
    x_sub_volumes_idx = []
    y_sub_volumes_idx = []
    z_sub_volumes_idx = []
    for x_idx in range(int(cell_prob_dataset.shape[0]/sub_vol_x)):
        for y_idx in range(int(cell_prob_dataset.shape[1]/sub_vol_y)):
            for z_idx in range(int(cell_prob_dataset.shape[2]/sub_vol_z)):
                x_next_idx = []
                x_next_idx.append(((sub_vol_x * x_idx), (sub_vol_x + sub_vol_x * x_idx)))
                x_sub_volumes_idx.append(x_next_idx)
                
                y_next_idx = []
                y_next_idx.append(((sub_vol_y * y_idx), (sub_vol_y + sub_vol_y * y_idx)))
                y_sub_volumes_idx.append(y_next_idx)
                
                z_next_idx = []
                z_next_idx.append(((sub_vol_z * z_idx), (sub_vol_z + sub_vol_z * z_idx)))
                z_sub_volumes_idx.append(z_next_idx)
    return x_sub_volumes_idx, y_sub_volumes_idx, z_sub_volumes_idx

def cell_detect_big_data_mpi():
    """
    Volume is divided into several sub-volumes and a different rank detects cells within a few sub-volumes. 
    Detected cells coordinates and labeled cell map is written into the data set of a HDF5 file by each rank.
    Parameters
    ----------
    Input: The volume cell probability map file location is specified in the segmentation_param.py file.
    
    Returns
    ------
    Cell centroids are written into a new data set created in the input file.
    Labeled cell maps are written into a new data set created within the input file.
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
    if not hdf_file.get('volume_cell_probability_map', getclass=True):
        print("**** 'volume_cell_probability_map' *** data set does not exist")
        return
    
    cell_prob_dataset = hdf_file['volume_cell_probability_map']
    print("cell_prob_dataset.shape", cell_prob_dataset.shape)
    # Delete "Volume Cell Map" and "Volume Centroids" data sets if they exists.
    if hdf_file.get('Volume Cell Map', getclass=True):
        hdf_file.__delitem__('Volume Cell Map')
    if hdf_file.get('Volume Centroids', getclass=True):
        hdf_file.__delitem__('Volume Centroids')
    
    # Divide the volume into sub-volumes. Compute start and end indices for each sub-volume
    x_sub_volumes_idx, y_sub_volumes_idx, z_sub_volumes_idx = compute_sub_volumes(cell_prob_dataset)
    print("Done with computing sub-volumes - This is rank %d of %d running on %s" % (rank, size, name))
    iterations = int(len(x_sub_volumes_idx) / size)
    partial_iterations = int(len(x_sub_volumes_idx) % size)
    
    # Create Data Set for the whole volume cell map
    vol_cell_map = hdf_file.create_dataset("Volume Cell Map", np.shape(cell_prob_dataset), dtype='uint32')
    vol_centroids_sz = iterations * size * max_no_cells
    # Create Data Set for Centroids detected in the volume
    vol_centroids_ds = hdf_file.create_dataset("Volume Centroids", (vol_centroids_sz, 4), dtype='float32')
    print("Volume Centroids size is %d" % vol_centroids_sz)
    centroids_zeros = np.zeros((max_no_cells), dtype='float32')
    
    for idx in range(iterations):
        x_idx = x_sub_volumes_idx[rank + (size * idx)]
        y_idx = y_sub_volumes_idx[rank + (size * idx)]
        z_idx = z_sub_volumes_idx[rank + (size * idx)]
        
        cell_prob_map = cell_prob_dataset[x_idx[0][0] : x_idx[0][1], y_idx[0][0] : y_idx[0][1], 
                                          z_idx[0][0] : z_idx[0][1]]
        print("***Cell Sub-volume*** to be processed by rank %d x, y, z  %d:%d, %d:%d, %d:%d" % 
              (rank, x_idx[0][0], x_idx[0][1], y_idx[0][0], y_idx[0][1], z_idx[0][0], z_idx[0][1]))
        
        centroids, cell_map = detect_cells(cell_prob_map, cell_probability_threshold,
                                           stopping_criterion, initial_template_size,
                                           dilation_size, max_no_cells)
        # The below line needs more work - if cell_map > 2GB it will not work.
        vol_cell_map[x_idx[0][0] : x_idx[0][1], y_idx[0][0] : y_idx[0][1],
                     z_idx[0][0] : z_idx[0][1]] = cell_map
        
        # The sub-volume indices need to be set to the whole volume indices.
        centroids[:, 0] = centroids[:, 0] + x_idx[0][0]
        centroids[:, 1] = centroids[:, 1] + y_idx[0][0]
        centroids[:, 2] = centroids[:, 2] + z_idx[0][0]
        # Below code may need more work - if the assumption of a pre-defined number cells to be detected 
        # in a sub-volume is not good.
        # Pad centroids with zeroes if found less than max_no_cells.
        if len(centroids) < max_no_cells:
            centroids = np.pad(centroids, pad_width=((0, (max_no_cells - len(centroids))), (0,0)), 
                               mode='constant', constant_values=0)
        # Save centriods into the data set slot
        cent_idx = rank + size * idx
        vol_centroids_ds[cent_idx * max_no_cells : (cent_idx + 1) * max_no_cells,:] = centroids
    
    hdf_file.close()
    # Ignore the partial iteration for now - i.e. assumes the number of sub-volumes is multiple of size.
    print("Done with Cell Detection and Number of sub-volumes ignored is %d" % partial_iterations)

if __name__ == '__main__':
    cell_detect_big_data_mpi()
