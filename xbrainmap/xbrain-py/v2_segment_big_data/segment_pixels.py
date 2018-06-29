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

'''
Segments pixels in a composite input image dataset into multiple image datasets.
'''

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
__all__ = ['segment_pixels']

def segment_pixels():
    """ 
    Separates pixels in the input sub-volume image files and creates an hdf5 file for each input file.
    Pixel mask for each class defined in the trained data is given in a dataset of a hdf5 file
    for each sub-volume. Also, composite sub-volume image for each sub-volume is given in another hdf5
    file. This script creates segmented image for each sub-volume by element by element multiplication 
    of the mask and the corresponding composite sub-volume image. Segmented pixels for each class of 
    the sub-volume is written into a separated dataset of the segmented output hdf5 file/sub-volume.
    

    Inputs: 
    Composite sub-volumes image hdf5 files.
    Pixel mask for each class/label in a hdf5 file per sub-volume.
    
    Outputs:
    a hdf5 file per sub-volume with a dataset for each defined segmented class.
    """
    
    start_time = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    name = MPI.Get_processor_name()
    
    ilastik_classes = get_ilastik_labels()
    if rank == 0:
        print("*** Entered segment_pixels(), Time is %d and size is %d ****" % (time.time(), size))
        print("Ilastik classes are ", ilastik_classes)
    
    # Get list of all files with probability maps (mask dataset) for pixel classes. 
    mask_files = sorted(glob(hdf_subvol_files_location + '/*Probabilities.h5'))
    if not mask_files:
        print("*** Did not find any file ending with .h5 extension  ***", hdf_subvol_files_location)
        return
    im_files = sorted(glob(hdf_subvol_files_location + '/*.hdf5'))
    if len(mask_files) != len(im_files):
        print("Number of mask files is %d and image files is %d and not the same" % (len(mask_files), len(im_files)))
        return
    
    if rank == 0:
        print("Number of HDF5 files is %d, and Number of processes is %d" % ((len(mask_files)), size))
        print("Sub-Volume file location is %s" % outimage_file_location)
        # Remove sub-volume files from previous run
        if os.path.exists(outimage_file_location):
            subvolfiles = glob(outimage_file_location + '/subvol*.h5')
            for subfile in subvolfiles:
                print("*** Removing file ***", subfile)
                os.remove(subfile)
            
        # Create directory for output pixel maps if it does not exist.
        if not os.path.exists(outimage_file_location):
            print("*** Creating directory %s ***" % outimage_file_location)
            os.mkdir(outimage_file_location)
    
    comm.Barrier()
    
    iterations = int(len(mask_files) / size) + (len(mask_files) % size > 0)
    for idx in range(iterations):
        if (rank + (size * idx)) >= len(mask_files):
            print("\nBREAKING out, my rank is %d, number of files is %d, size is %d and idx is %d" %
                  (rank, len(mask_files), size, idx))
            break
        im_read_start = time.time()
        im_file = h5py.File(im_files[rank + (size * idx)], 'r')
        im_datasets = []
        for ds in im_file.keys():
            im_datasets.append(ds.encode('ascii'))
        if rank == 0:
            print("Datasets in this file are %s: " % (im_datasets))
            print("Input Image file name is %s: " % (im_files[rank + (size * idx)]))
        
        # Retrieve the indices into whole volume for this sub-volume.
        orig_idx_ds = im_file[im_datasets.pop(im_datasets.index('orig_indices'))]
        print("file datasets are", im_datasets)
        orig_idx_data = orig_idx_ds[...]
        
        # Retrieve overlap size to the right side of the sub-volume.
        rightoverlap_ds = im_file[im_datasets.pop(im_datasets.index('right_overlap'))]
        print("file datasets are", im_datasets)
        rightoverlap_data = rightoverlap_ds[...]
        
        # Retrieve overlap size to the left side of the sub-volume.
        leftoverlap_ds = im_file[im_datasets.pop(im_datasets.index('left_overlap'))]
        print("file datasets are", im_datasets)
        leftoverlap_data = leftoverlap_ds[...]
        im_ds = im_file[im_datasets[0]]
        im_data = im_ds[...]
        print("Input Image Shape is", im_data.shape)
        print("Sub-Volume indices are", orig_idx_data)
        print("Sub-Volume rightoverlap are", rightoverlap_data)
        print("Sub-Volume leftoverlap are", leftoverlap_data)
        im_file.close()
        if rank == 0:
            print("Image read time is %d Sec" % (time.time() - im_read_start))
        mk_file = h5py.File(mask_files[rank + (size * idx)], 'r')
        print("Mask file name is %s: " % (mask_files[rank + (size * idx)]))
        par, name = os.path.split(outimage_file_location)
        # Create file for the sub-volume pixel map for the object class
        filenumber = str(rank + size * idx).zfill(5)
        im_out_filename = outimage_file_location + '/subvol_' + name + filenumber + '.h5'
        imout_file = h5py.File(im_out_filename, 'w')
        print("Output segmented objects file name is %s" % (im_out_filename))
        # Save sub-volume indices
        subvol_indx = imout_file.create_dataset('orig_indices', (6,), dtype='uint64')
        subvol_indx[...] = orig_idx_data
        # Save sub-volume right and left side overlaps.
        subvol_rightoverlap = imout_file.create_dataset('right_overlap', (3,), dtype='uint8')
        subvol_rightoverlap[...] = rightoverlap_data
        subvol_leftoverlap = imout_file.create_dataset('left_overlap', (3,), dtype='uint8')
        subvol_leftoverlap[...] = leftoverlap_data
        for label in range(len(ilastik_classes)):
            # Get mask dataset for the object class 
            if rank == 0:
                print("Output segmented objects file name is %s and dataset name is %s" % 
                      (im_out_filename, ilastik_classes[label]))
            mk_ds = mk_file[ilastik_classes[label]]
            mk_data = mk_ds[...]
            im_output_ds = imout_file.create_dataset(ilastik_classes[label], mk_data.shape, mk_data.dtype)
            multiply_time = time.time()
            im_output_ds[...] = im_data * mk_data
            if rank == 0:
                print("Sub-Volume IM output dataset name is and shape is", ilastik_classes[label], im_output_ds)
                print("Multiply time for one dataset is %d Sec" % (time.time() - multiply_time))
        mk_file.close()
        imout_file.close()
    end_time = time.time()
    print("Exec time is %d Sec, rank is %d" % ((end_time - start_time), rank))
    
if __name__ == '__main__':
    segment_pixels()
