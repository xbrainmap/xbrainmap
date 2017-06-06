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
import pdb

__author__ = "Mehdi Tondravi"
__copyright__ = "Copyright (c) 2017, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['ilastik_classify_mpi']

def ilastik_classify_mpi():
    """
    Divides many *.hdf5 sub-volume image files which are crteated from tiff files and calls Ilastik pixel
    classifier on each of *.hdf5 file.
    
    Inputs: 
    The *.hdf5 sub-volume files location is specified in seg_user_param.py file.
    Ilastik trained data - file location is specified in seg_user_param.py file.
        
    Output: 
    The *.h5 files (which have probability maps) location is specified in seg_user_param.py file.  
    
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    name = MPI.Get_processor_name()
    start_time = int(time.time())
    # Allow Ilatisk to use all available threads of the server/compute node.
    threads = int(no_of_threads/1)
#    threads = 1
    # Allow Ilastik to use available memory of the server/compute node.
    ram = ram_size
#    ram = int(ram_size/12)
    if rank == 0:
        print("*** size is %d, No of thread is %d, ram size is %d" % (size, threads, ram))
    # assumes sub-volume image file extension is .hdf5
    input_files = sorted(glob(hdf_subvol_files_location + '/*.hdf5'))
    if not input_files:
        print("*** Did not find any file ending with .hdf5 extension  ***")
        return
    # Delete existing files created by ilastik (*.h5 files).
    if rank == 0:
        print("Ilastik input files/hdf_files_location", hdf_subvol_files_location)
        oldoutput_files = sorted(glob(hdf_subvol_files_location + '/*.h5'))
        for file in oldoutput_files:
            print("*** Removing old Ilastik created file %s ***" % file)
            os.remove(file)
    
    comm.Barrier()
    
    data_sets = []
    indices_ds = []
    rightoverlap_ds = []
    leftoverlap_ds = []
    # Get the dataset name in each sub-volume file. Dataset name is the same as file name.
    # Convert from unicode to ASCII since Ilastik does not like unicode
    for file in input_files:
        f = h5py.File(file, 'r')
        name, ext = os.path.splitext(os.path.basename(file))
        data_sets.append((file + '/' + name).encode('ascii'))
        indices_ds.append(f['orig_indices'][...])
        rightoverlap_ds.append(f['right_overlap'][...])
        leftoverlap_ds.append(f['left_overlap'][...])
        f.close()
    
    if rank == 0:
        print("Number of input/HDF5 files is %d, and Number of processes is %d" % ((len(data_sets)), size))
    
    # Figure out how many sub-volume files each rank should handle.
    iterations = int(len(data_sets) / size) + (len(data_sets) % size > 0)
    # Divide pixel classification of sub-volume files among processes/ranks. 
    for idx in range(iterations):
        if (rank + (size * idx)) >= len(data_sets):
            print("\nBREAKING out, this rank is done with its processing, my rank is %d, number of files is %d, size is %d and idx is %d" %
                  (rank, len(data_sets), size, idx))
            break
        start_loop_time = time.time()
        data_set_name = data_sets[(rank + size * idx)]
        start_classify_time = time.time()
        hdf_dataset_path = classify_pixel_hdf(data_set_name, classifier, threads, ram)
        end_classify_time = time.time()
        classify_time = end_classify_time - start_classify_time
        print("Exec time for classification is %d Sec, rank is %d, hdf_dataset_path is %s" % 
              (classify_time, rank, hdf_dataset_path))
        # Create a dataset and save indices of the sub-volume into the whole volume.
        filename, dataset = os.path.split(hdf_dataset_path[0])
        file = h5py.File(filename, 'r+')
        subvol_indx = file.create_dataset('orig_indices', (6,), dtype='uint64')
        subvol_indx[...] = indices_ds[(rank + size * idx)]
        
        # Save the overlap sizes.
        subvol_rightoverlap = file.create_dataset('right_overlap', (3,), dtype='uint8')
        subvol_rightoverlap[...] = rightoverlap_ds[(rank + size * idx)]
        
        subvol_leftoverlap = file.create_dataset('left_overlap', (3,), dtype='uint8')
        subvol_leftoverlap[...] = leftoverlap_ds[(rank + size * idx)]
        file.close()
        end_loop_time = time.time()
        file_classify_time = end_loop_time - start_loop_time
        print("Exec Time per classifying one file is %d Sec, read/write time is %d Sec and rank is %d" % 
              (file_classify_time, (file_classify_time - classify_time), rank))
    
    end_time = int(time.time())
    exec_time = end_time - start_time
    print("*** My Rank is %d, exec time is %d sec - Done with classifying pixels in sub-volume files ***" % (rank, exec_time))

if __name__ == '__main__':
    ilastik_classify_mpi()

