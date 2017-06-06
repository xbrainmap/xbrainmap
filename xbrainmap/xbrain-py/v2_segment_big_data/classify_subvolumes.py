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

import pdb
import numpy as np
import h5py
from mpi4py import MPI
import os.path
from glob import glob
import time
from segmentation_param import *

__author__ = "Mehdi Tondravi"
__copyright__ = "Copyright (c) 2017, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['classify_subvolumes']

def classify_subvolumes():
    """ 
    Assigns each pixels to the class with the highest probability value determined by Ilastik classifier.
    
    Ilastik returns probability map in the "exported_data" dataset with the dimensions of x,y,z 
    (pixel location) by N values and N is the number of classes (labels) defined in the Ilastik 
    trained data file. This script adds N datasets to each of the input hdf5 files. Dimension of
    each dataset is x,y,z (pixel location) and value of zero or one. The pixel location in the added
    datasets with the highest probability value is one, zero otherwise.
    
    Input: Ilastik sub-volumes pixel classified hdf5 files - file location is specified in seg_user_param.py.
    
    Output: Datasets added to the sub-volume input files - one dataset per each class/label in trained data.
    """
    
    start_time = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    name = MPI.Get_processor_name()
    
    ilastik_classes = get_ilastik_labels()
    if rank == 0:
        print("*** Time is %d Entered classify_subvolumes() ****" % time.time())
        print("Ilastik classes are ", ilastik_classes)
    
    # Get list of pixel classsified sub-volume files with probability maps . Assumes file extension is .h5
    input_files = sorted(glob(hdf_subvol_files_location + '/*Probabilities.h5'))
    if not input_files:
        print("*** Did not find any file ending with .h5 extension  ***", hdf_subvol_files_location)
        return
    
    if rank == 0:
        print("Number of HDF5 files is %d, and Number of processes is %d" % ((len(input_files)), size))
        print("Sub-Volume file location is %s" % hdf_subvol_files_location)
    iterations = int(len(input_files) / size) + (len(input_files) % size > 0)
    for idx in range(iterations):
        if (rank + (size * idx)) >= len(input_files):
            print("\nBREAKING out, my rank is %d, number of files is %d, size is %d and idx is %d" %
                  (rank, len(input_files), size, idx))
            break
        
        infile = h5py.File(input_files[rank + (size * idx)], 'r+')
        exds = infile[ilastik_ds_name]
        start_read = time.time()
        export_indata = exds[...]
        export_outdata = np.zeros(export_indata.shape, dtype='uint8')
        print("export_indata data shape is", export_indata.shape)
        end_read = time.time()
        print("Read time for all rows of Ilastik Prob map is %d Sec" % (end_read - start_read))
        print_cycle = 100
        start_loop_time = time.time()
        for row in range(export_outdata.shape[0]):
            for colmn in range(export_outdata.shape[1]):
                outdata = np.zeros((export_outdata.shape[2], export_outdata.shape[3]), 'uint8')
                if row == 0:
                    if colmn == 0:
                        print("outdata shape is", outdata.shape)
                outdata[np.arange(len(outdata)), np.argmax(export_indata[row, colmn, :, :], axis=-1)] = 1
                if np.count_nonzero(outdata) != export_outdata.shape[2]:
                    print("Something is very wrong - check row %d and colmn %d" % (row, colmn))
                export_outdata[row, colmn, :, :] = outdata.copy()
            if row % print_cycle == 0:
                print("time to classify %d rows is %d Sec" % (print_cycle, (time.time() - start_loop_time)))
                start_loop_time = time.time()
         # Segment classes and saved them into datasets
        start_ds_time = time.time()
        no_of_classes = export_outdata.shape[3]
        export_outdata = np.transpose(export_outdata, (3, 0, 1, 2))
        for idx in range(no_of_classes):
            time_per_ds = time.time()
            if infile.get(ilastik_classes[idx], getclass=True):
                print("*** Deleting subvolume object map %s ***" % (ilastik_classes[idx]))
                infile.__delitem__(ilastik_classes[idx])
            outdataset = infile.create_dataset(ilastik_classes[idx], 
                                               (export_outdata.shape[1], export_outdata.shape[2],
                                                export_outdata.shape[3]), export_outdata.dtype)
            outdataset[...] = export_outdata[idx, ...]
            print("Time to save %s dataset is %d Sec" % (ilastik_classes[idx], (time.time() - time_per_ds)))
        # Print time to save datasets
        print("Time to save all object maps is %d Sec" % (time.time() - start_ds_time))
        infile.close()
    print("Exec time for this function was %d Sec and rank is %d" % ((time.time() - start_time), rank))
    
if __name__ == '__main__':
    classify_subvolumes()
