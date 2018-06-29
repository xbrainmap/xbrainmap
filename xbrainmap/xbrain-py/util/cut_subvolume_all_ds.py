#!/usr/bin/env python

"""
Cut a sub-volume from the given file and dimensions of the cut for all datasets in the input file.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import h5py
import numpy as np
from glob import glob
import os.path
import time
import sys
import pdb

def cut_subvolume_all_ds():
    """
    Usage example:
    python cut_subvolume_all_ds.py /projects/project_abc/dataset_abc.h5 400:1200 800:1600 600:1800
    """
    start_time = int(time.time())
    arg_cnt = len(sys.argv)
    if arg_cnt < 5:
        print("Not enough parameters - must have file name, x, y and z sizes")
        return
    
    hdf_file_name = sys.argv[1]
    x_dim = map(int, sys.argv[2].split(':'))
    y_dim = map(int, sys.argv[3].split(':'))
    z_dim = map(int, sys.argv[4].split(':'))

    
    print("File name", hdf_file_name, x_dim, y_dim, y_dim)
    
    infile = h5py.File(hdf_file_name, 'r')
    ds_grp = []
    for grp in infile.keys():
        ds_grp.append(grp.encode('ascii'))
    print("Dataset names are %s" % ds_grp)
    directory, filename = os.path.split(hdf_file_name)
    par_directory, file_directory = os.path.split(directory)
    print("Directory is %s and input file name is %s" % (directory, filename))
    print("parent dir is %s and file directory is %s" % (par_directory, file_directory))
    outfilename = 'x' + str(x_dim[0]) + '_' + 'y' + str(y_dim[0]) + '_' + 'z' + str(z_dim[0]) + '_' + filename
    print("Output file name is %s" % outfilename)
    out_dir = par_directory + '/cuts_' + file_directory
    print("output file directory is %s" % out_dir)
    if not os.path.exists(out_dir):
        print("Creating directory %s" % out_dir)
        os.mkdir(out_dir)
    outfile = h5py.File((out_dir+'/'+outfilename), 'w')
    for idx in range(len(ds_grp)):
        dataset = infile[ds_grp[idx]]
        mydata = dataset[x_dim[0]:x_dim[1], y_dim[0]:y_dim[1], z_dim[0]:z_dim[1]]
        print("Dataset name is %s" % ds_grp[idx])
        print("Cut data shape, dtype and input data shape are ", mydata.shape, mydata.dtype, dataset.shape)
        outds = outfile.create_dataset(ds_grp[idx], mydata.shape, dtype=mydata.dtype)
        outds[...] = mydata
    outfile.close()
    infile.close()

if __name__ == '__main__':
    cut_subvolume_all_ds()



