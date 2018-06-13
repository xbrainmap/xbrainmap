#!/usr/bin/env python

"""
Converts hdf5 file image slices into tiff image files vi multiple/MPI  processes/ranks
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import h5py

import numpy as np
from skimage.io import imsave
from glob import glob
import os.path
from mpi4py import MPI
import time
import sys
import pdb

__author__ = "Mehdi Tondravi"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['hdf5_to_tiff']

def util_hdf5_to_tiff_mpi():
    
    """
    Usage example:
    1) make tiff files from the first dataset found in the specified hdf5 file.
    mpirun -np 4 python util_hdf5_to_tiff_mpi.py /projects/project_abc/dataset_abc.h5
    
    2) make tiff files from the specified hdf5 file and dataset.
    mpirun -np 4 python util_hdf5_to_tiff_mpi.py /projects/project_abc/dataset_abc.h5  "Cell_Body"
    
    3) make tiff files from the specified hdf5 file, dataset, start and stop slices.
    mpirun -np 4 python util_hdf5_to_tiff_mpi.py /projects/project_abc/dataset_abc.h5  "Cell_Body" 100 400
    
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    name = MPI.Get_processor_name()
    if rank == 0:
        print("Usage: python util_hdf5_to_tiff.py  hdf5_filename optional: dataset name")
    
    start_time = int(time.time())
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
    if rank == 0:
        print("Input HDF5 file name is %s, and input dataset name is %s" % (hdf_file_name, dataset_name))
    
    hfile = h5py.File(hdf_file_name, 'r', driver='mpio', comm=comm)
    hfile_grp = []
    for grp in hfile.keys():
        hfile_grp.append(grp.encode('ascii'))
    if rank == 0:
        print("Dataset group name in this file are : ", hfile_grp)
    if dataset_name:
        ds_name = dataset_name
    else:
        ds_name = hfile_grp[0]
    if rank == 0:
        print("Input File is name is %s and dataset name is %s" % (hdf_file_name, ds_name))
        print("size is %d" % size)
    input_ds = hfile[ds_name]
    if rank == 0:
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
    ds_name = str(ds_name)
    dir, input_filename = os.path.split(hdf_file_name)
    par_dir, dir_name = os.path.split(dir)
    tiff_files_dir = par_dir + '/' + dir_name + '_' + ds_name +'_tiff'
    base_name, file_ext = os.path.splitext(input_filename)
    base_name = base_name + ds_name
    # Remove all *.tiff files from previous runs. Create directory if does not exist
    if rank == 0:
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
    comm.Barrier()
    
    iteration = int(input_ds.shape[0] / size) + (input_ds.shape[0] % size > 0)
    count = 0
    for idx in range(iteration):
        if (rank + (size * idx)) >= input_ds.shape[0]:
            print("\nBREAKING out, my rank is %d and idx is %d" % (rank, idx))
            break
        row = rank + size * idx
        tiff_file = tiff_files_dir + '/' + base_name + '_' + str(row).zfill(5) + '.tiff'
        if row == 0:
            print("tiff file name is", tiff_file)
        tiff_data = input_ds[row, :, :]
        if row == 0:
            print("tiff_data shape and data type are ", tiff_data.shape, tiff_data.dtype)
        imsave(tiff_file, tiff_data, plugin='tifffile')
        if row != 0 and row % 100 == 0:
            print("wrote file for slice %d out of %d slices" % (row, input_ds.shape[0]))
        
    hfile.close()
    end_time = time.time()
    exec_time = end_time - start_time
    print("Done dividing tiff files, rank is %d, size is %d, name is %s, exec time is %d sec" % (rank, size, name, exec_time))

if __name__ == '__main__':
    util_hdf5_to_tiff_mpi()

