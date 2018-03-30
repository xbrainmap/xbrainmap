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
Divides tiff files into many stacks and call Ilastik to classify each tiff stack by a rank.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path
import h5py
import numpy as np
from glob import glob
from mpi4py import MPI
import time
from classify_pixel_hdf import classify_pixel_hdf
from segmentation_param import *

__author__ = "Mehdi Tondravi"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['ilastik_classify_mpi']

def ilastik_classify_mpi():
    """
    Divides many *.hdf5 files which are crteated from tiff files and calls Ilastik classifier on each 
    *.hdf5 file.
    
    Parmaters
    ---------
    Input: The *.hdf5 files (tiff images) location is specified in the segmentation_param.py file. 
    
    Output: The *.h5 files (which have probability maps) location is specified in the segmentation_param.py file.  
    
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    name = MPI.Get_processor_name()
    
    start_time = int(time.time())
#    threads = int(no_of_threads/1)
    threads = 1
    ram = int(ram_size/no_of_threads)
    
    # assumes file extension is .hdf5
    hdf5_files = sorted(glob(hdf_files_location + '/*.hdf5'))
    if not hdf5_files:
        print("*** Did not find any file ending with .hdf5 extension  ***")
        return
    # Delete existing files created by ilastik (*.h5 files).
    print("hdf_files_location", hdf_files_location)
    if rank == 0:
        h5_files = sorted(glob(hdf_files_location + '/*.h5'))
        for file in h5_files:
            os.remove(file)
    
    comm.Barrier()
    
    data_sets = []
    vol_shape = np.zeros((1,3), dtype='uint64')
    # Get the data set name in each file, there is only one data set per file. 
    # Convert from unicode to ASCII since Ilastik does not like unicode
    for file in hdf5_files:
        f = h5py.File(file, 'r')
        ds = [x for x in f][0]
        data_sets.append((file + '/' + ds).encode('ascii'))
        f.close()
    
    files_per_rank = int(len(data_sets)/size)
    files_not_in_rank_list = data_sets[(files_per_rank * size) : len(data_sets)]
    
    print("Number of HDF5 files is %d, and Number of processes is %d" % ((len(data_sets)), size))
    print("Number of files not processed is %d" % (len(files_not_in_rank_list)))
    
    # Divide classification among processes/ranks. 
    process_data_sets = []
    
    for idx in range(files_per_rank):
        # Process_data_sets.append(data_sets[(rank + size * idx)])
        data_set_name = data_sets[(rank + size * idx)]
        hdf_dataset_path = classify_pixel_hdf(data_set_name, classifier, threads, ram)
        print("hdf_dataset_path is %s and my rank is %d" % (hdf_dataset_path, rank))
        
        # Create cell and vessel probability map data sets.
        file, dataset = os.path.split(hdf_dataset_path[0])
        file = h5py.File(file, 'r+')
        probability_maps = file[dataset]
        ar_shape = probability_maps.shape[0:-1]
        print("probability_maps.shape", probability_maps.shape)
        cell_prob_map = file.create_dataset("cell_probability_map", ar_shape, dtype='float32')
        cell_prob_map[...] = probability_maps[:, :, :, cell_label_idx]
        print("cell_prob_map.shape and my rank is ", cell_prob_map.shape, rank)
        
        vessel_prob_map = file.create_dataset("vessel_probability_map", ar_shape, dtype='float32')
        vessel_prob_map[...] = probability_maps[:, :, :, vessel_label_idx]
        print("vessel_prob_map.shape and my rank is " , vessel_prob_map.shape, rank)
        file.close()
    
    comm.Barrier()
    
    end_time = int(time.time())
    exec_time = end_time - start_time
    print("*** My Rank is %d, exec time is %d sec - Done with classifying cells & vessels ***" % (rank, exec_time))

if __name__ == '__main__':
    ilastik_classify_mpi()

