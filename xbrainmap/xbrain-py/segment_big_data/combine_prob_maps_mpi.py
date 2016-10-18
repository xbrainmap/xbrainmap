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
Combines many *.h5 files with cell & vessels probability maps into one HDF5 file. 
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path
import h5py
import numpy as np
from glob import glob
from mpi4py import MPI
from segmentation_param import *

__author__ = "Mehdi Tondravi"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['combine_prob_maps_mpi']


def combine_prob_maps_mpi():
    """
    Combines many *.h5 files created for sub-volumes with cell & vessel probability maps into one big
    file. To speed this up many ranks/processes are created to write maps for sub-volumes into one whole
    volume/file.
    
    Parameters 
    ----------
    Input: The sub-volume probability maps file location is specified in the segmentation_param.py file. 
    
    Output: The whole volume probability maps file location is specified in the segmentation_param.py file. 
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    name = MPI.Get_processor_name()
    
    # Get list of all files with cell & vessel probability maps. Assumes file extension is .h5
    hdf5_files = sorted(glob(hdf_files_location + '/*.h5'))
    if not hdf5_files:
        print("*** Did not find any file ending with .hdf5 extension  ***", hdf_files_location)
        return
    # Shape/Dimention of probability maps in all files are the same. Get the shape from one/first file.
    cell_file_shape = np.zeros((1,3), dtype='uint64')
    vessel_file_shape = np.zeros((1,3), dtype='uint64')
    f = h5py.File(hdf5_files[0], 'r')
    if f.get('cell_probability_map'):
        cell_file_shape = np.array(f['cell_probability_map'].shape)
    if f.get('vessel_probability_map'):
        vessel_file_shape = np.array(f['vessel_probability_map'].shape)
    f.close()
    
    files_per_rank = int(len(hdf5_files)/size)
    # If number of HDF5 files is not exact multiple of ranks then the files in below will not be used.
    files_not_in_rank_list = hdf5_files[(files_per_rank * size) : len(hdf5_files)]
    hdf5_files = hdf5_files[0: (files_per_rank * size)]
    files_for_rank = hdf5_files[(rank * files_per_rank) : ((rank + 1) * files_per_rank)]
    
    # Create an hdf file to contain the whole volume cell & vessel probability maps.
    # This file name is made of concatenating first and last file name plus 'volume_prob'.
    first_file_name, first_file_ext = os.path.splitext(os.path.basename(hdf5_files[0]))
    last_file_name, last_file_ext = os.path.splitext(os.path.basename(hdf5_files[(files_per_rank * size -1)]))
    print("Number of HDF5 files is %d, and Number of processes is %d" % ((len(hdf5_files)), size))
    print("Number of files not process is %d" % (len(files_not_in_rank_list)))
    vol_prob_map_file = (volume_map_file_location + '/' + first_file_name + '_' + 
                         last_file_name + '_volume_prob_map' + '.h5')
    
    # Compute the volume cell & vessel probability map shapes.
    cell_vol_shape = np.copy(cell_file_shape)
    vessel_vol_shape = np.copy(vessel_file_shape)
    cell_vol_shape[0] = cell_file_shape[0] * len(hdf5_files)
    vessel_vol_shape[0] = vessel_file_shape[0] * len(hdf5_files)
    
    # Create directory for the whole volume probability maps if it does not exists.
    if rank == 0:
        if not os.path.exists(volume_map_file_location):
            os.mkdir(volume_map_file_location)
            print("File directory for whole volume probability maps did not exist, was created")
        else:
            # Delete existing combined file if exsits.
            combined_files = sorted(glob(volume_map_file_location + '/*_volume_prob_map.h5'))
            if combined_files:
                for file in combined_files:
                    os.remove(file)
                
            
    comm.Barrier()
    
    vol_map_file = h5py.File(vol_prob_map_file, 'w', driver='mpio', comm=comm)
    vol_cell_prob_map = vol_map_file.create_dataset('volume_cell_probability_map', cell_vol_shape, dtype='float32')
    vol_vessel_prob_map = vol_map_file.create_dataset('volume_vessel_probability_map', vessel_vol_shape, dtype='float32')
    
    for idx in range(len(files_for_rank)):
        f = h5py.File(files_for_rank[idx], 'r+')
        first_idx = (idx + rank * files_per_rank) * cell_file_shape[0]
        last_idx = (idx + 1 + rank * files_per_rank) * cell_file_shape[0]
        print("first and last idx are", first_idx, last_idx)
        if f.get('cell_probability_map'):
            cell_ds= f['cell_probability_map']
            for i in range(last_idx - first_idx):
                vol_cell_prob_map[first_idx+i : first_idx+i+1,:,:] = cell_ds[i,:,:]
        
        if f.get('vessel_probability_map'):
            vessel_ds =f['vessel_probability_map']
            for i in range(last_idx - first_idx):
                vol_vessel_prob_map[first_idx+i : first_idx+i+1,:,:] = vessel_ds[i,:,:]
        
        f.close()
    vol_map_file.close()
    print("*** My Rank is %d - Done with combining cells & vessels prob maps  ***" % rank) 
    print("My Rank, cells prob shape & vessel prob shape are", rank, cell_vol_shape, vessel_vol_shape)

if __name__ == '__main__':
    combine_prob_maps_mpi()

