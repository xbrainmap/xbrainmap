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
Converts reconstructed tiff image file into hdf5.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import h5py
import numpy as np
from skimage.io import imread
from glob import glob
import os.path
import time
import sys
import pdb

__author__ = "Mehdi Tondravi"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['tiff_to_hdf5_files']

def util_tiff_to_hdf5():
    """
    Create a directory to hold the hdf file at same level as tiff files directory. This directory name
    is the same as tiff directory name plus "_hdf". Create an hdf5 file with the name made of 
    the concatenation of the first tiff file and last tiff file. Create an hdf data set name with the 
    same name as the directory name of the tiff files.
    As an example if the tiff files are located at, "/path/Downloads/eva_block" directory,
    the first tiff name is "data_00860.tiff.tif" and the last tiff file name is "data_01139.tiff.tif" 
    then the hdf file created is:
    /path/Downloads/eva_block_hdf/data_00860.tiff_data_01139.tiff.hdf5
    and hdf5 data set name is "eva_block".
    
    Input: Tiff files location
    Output: HDF5 file 
    
    """
    start_time = int(time.time())
    arg_cnt = len(sys.argv)
    if arg_cnt < 2:
        print("must give TIFF files location")
        return
    elif arg_cnt == 2:
        tiff_files_location = sys.argv[1]
    elif arg_cnt > 2:
        print("too many arguements")
        return
    
    files = sorted(glob(tiff_files_location + '/*.tif*'))
    parent_dir, tiff_dir = os.path.split(tiff_files_location)
    hdf_dir = parent_dir + '/' + tiff_dir + '_' + 'tiff' + '_hdf'
    
    # Remove all *.hdf5 files from previous runs. Create directory if does not exist
    print("**** File location is ****", hdf_dir)
    if os.path.exists(hdf_dir):
        hdf5_files =  glob(hdf_dir + '/*.hdf5')
        for file in hdf5_files:
            print("*** Removing file ***", file)
            os.remove(file)
    if not os.path.exists(hdf_dir):
        print("*** Creating directory ***", hdf_dir)
        os.mkdir(hdf_dir)
    
    data_shape = imread(files[0]).shape
    data_type = imread(files[0]).dtype
    print("*** Number of files is %d ***" % (len(files)))
    first_file_name, first_file_ext = os.path.splitext(os.path.basename(files[0]))
    last_file_name, last_file_ext = os.path.splitext(os.path.basename(files[-1]))
    hdf_file_name = hdf_dir + '/'+first_file_name + '_' + last_file_name + '.hdf5'
    
    hdf_file = h5py.File(hdf_file_name, 'w')
    data_set_name = tiff_dir
    print("*** Dataset name is ***", data_set_name)
    data_set = hdf_file.create_dataset(data_set_name, (len(files), data_shape[0], data_shape[1]), data_type)
    for idx in range(len(files)):
        imread_start = time.time()
        imarray = imread(files[idx], plugin='tifffile')
        data_set[idx, :, :] = imarray
        imread_end = time.time()
        if idx !=0 and idx % 100 == 0:
            print("idx is %d, time for read is %d sec, file is %s" % 
                  (idx, (imread_end - imread_start), files[idx]))
    
    print("data shape is", data_set.shape)
    hdf_file.close()
    end_time = int(time.time())
    exec_time = end_time - start_time
    print("Done dividing tiff files, exec time is %d sec" % (exec_time))

if __name__ == '__main__':
    util_tiff_to_hdf5()
