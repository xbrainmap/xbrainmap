#!/usr/bin/env python

"""
Converts hdf5 file image slices into tiff image files.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import h5py
import numpy as np
from skimage.io import imsave
from glob import glob
import os.path
import time
import sys
import pdb

__author__ = "Mehdi Tondravi"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['hdf5_to_tiff']

def util_hdf5_to_tiff():
    
    """
    Usage example:
    1) make tiff files from the first dataset found in the specified hdf5 file.
    python util_hdf5_to_tiff.py /projects/project_abc/dataset_abc.h5
    
    2) make tiff files from the specified hdf5 file and dataset.
    python util_hdf5_to_tiff.py /projects/project_abc/dataset_abc.h5  "Cell_Body"
    
    3) make tiff files from the specified hdf5 file, dataset, start and stop slices.
    python util_hdf5_to_tiff.py /projects/project_abc/dataset_abc.h5  "Cell_Body" 100 400
    
    """
    start_time = int(time.time())
    print("Usage: python util_hdf5_to_tiff.py  hdf5_filename optional: dataset name, begin slice , end slice")
    
    start_row = -1
    end_row = -1
    arg_cnt = len(sys.argv)
    if arg_cnt < 2:
        print("must give hdf5 file name")
        return
    elif arg_cnt == 2:
        hdf_file_name = sys.argv[1]
        dataset_name = []
    elif arg_cnt == 3:
        hdf_file_name = sys.argv[1]
        dataset_name = sys.argv[2]
    elif arg_cnt == 4:
        hdf_file_name = sys.argv[1]
        dataset_name = sys.argv[2]
        start_row = int(sys.argv[3])
    elif arg_cnt == 5:
        hdf_file_name = sys.argv[1]
        dataset_name = sys.argv[2]
        start_row = int(sys.argv[3])
        end_row = int(sys.argv[4])
    
    if not os.path.isfile(hdf_file_name):
        print("File %s does not exist" % hdf_file_name)
        return
    print("Input HDF5 file name is %s, and input dataset name is %s" % (hdf_file_name, dataset_name))
    
    hfile = h5py.File(hdf_file_name, 'r')
    hfile_grp = []
    for grp in hfile.keys():
        hfile_grp.append(grp.encode('ascii'))
    print("Dataset group name in this file are : ", hfile_grp)
    if dataset_name:
        ds_name = dataset_name
    else:
        ds_name = hfile_grp[0]
    print("Input File name is %s and dataset name is %s" % (hdf_file_name, ds_name))
    input_ds = hfile[ds_name]
    print("*** Input Data Shape is ***", input_ds.shape)
    
    if start_row == -1:
        start_row = 0
    if end_row == -1:
        end_row = input_ds.shape[0]
    print("Start slice is %d, End slice is %d" %(start_row, end_row))
    if end_row > input_ds.shape[0]:
        print("*** End Slice out of range ***")
        return
    if start_row < 0:
        print("*** End Slice out of range ***")
        return
    dir, input_filename = os.path.split(hdf_file_name)
    par_dir, dir_name = os.path.split(dir)
    ds_name = str(ds_name)
    tiff_files_dir = par_dir + '/' + dir_name + '_' + ds_name +'_tiff'
    base_name, file_ext = os.path.splitext(input_filename)
    base_name = base_name + ds_name
    # Remove all *.tiff files from previous runs. Create directory if does not exist
    print("**** Tiff Files Directory is ****", tiff_files_dir)
    if os.path.exists(tiff_files_dir):
        old_files =  glob(tiff_files_dir + '/*.tiff')
        if old_files:
            print("Removing Old *.tiff files")  
            for file in old_files:
                os.remove(file)
    if not os.path.exists(tiff_files_dir):
        print("*** Creating directory ***", tiff_files_dir)
        os.mkdir(tiff_files_dir)
    
    count = 0
    for row in range(start_row, end_row):
        tiff_file = tiff_files_dir + '/' + base_name + '_' + str(row).zfill(5) + '.tiff'
        if row == 0:
            print("tiff file name is", tiff_file)
        tiff_data = input_ds[row, :, :]
        if row == 0:
            print("tiff_data shape and data type are ", tiff_data.shape, tiff_data.dtype)
        imsave(tiff_file, tiff_data, plugin='tifffile')
        if row != 0 and row % 500 == 0:
            print("wrote file for slice %d out of %d slices" % (row, input_ds.shape[0]))
        
    hfile.close()
    end_time = time.time()
    exec_time = end_time - start_time
    print("Done dividing tiff files, exec time is %d sec" % (exec_time))

if __name__ == '__main__':
    util_hdf5_to_tiff()

