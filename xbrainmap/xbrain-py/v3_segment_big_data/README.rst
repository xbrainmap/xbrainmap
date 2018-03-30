=======================================================
Workflow for segmentation of Terabyte sized 3D Datasets
=======================================================

Workflow Description
--------------------
3D imaging of brain at single-neuron resolution generates Terabyte sized datasets. In follow-on analysis, it is essential to go beyond a 3D density image to segment out features of interest (such as blood vessels, cells, and axons in the example of brains).

Scripts in this folder contains a parallel processing workflow for segmenting large/Terabyte sized datasets based on Ilastik Pixel Classification Workflow.

Workflow Architecture
---------------------
**This workflow consists of the following processing steps:**

**\1. Creating Sub-Volumes for Automated Segmentation**

The input to this workflow is TIFF file stack in grayscale. The first step is to create an HDF5 file with a dataset configured to correspond the entire 3D volume. The 3D volume/dataset then is divided into several overlapping subarrays to be pixel classified by the next step. Each subarray is saved into an HDF5 file. 
This step should be run on a set of networked servers to speed up the processing.

**\2. Automated Segmentation with Parallelized Ilastik**

In this step, Ilastik pixel classification process is run on each subarray. Input to each Ilastik classifier process is the trained data file and a subarray/sub-volume file from previous step. Ilastik classifier creates K probability maps and K is the number of annotated voxel classes/types in the trained data file. Then each pixel in the probability map is assigned to the class with the highest probability value. Output from this step is an HDF5 file with K subarray/sub-volume datasets for each input subarray file.
This step should be run on a set of networked servers to speed up the processing.

**\3. Merging of overlapping sub-volumes**

In this step, the K subarrays in sub-volume files are combined to create K arrays for the volume. 
This step should be run on a set of networked servers to speed up the processing.


Configuring python environments to run this workflow
----------------------------------------------------

To read and write into a common HDF5 file need hdf5 package compiled in “parallel” mode. Ilastik is not built with this option and will be too much effort to build it for parallel HDF5. So, should create the following two python environments and take advantage of the pre-built required Ilastik packages.

**\1. Python Environment for making and combining sub-volumes**

This environment should be used when creating sub-volumes for segmentation, and when combining segmented sub-volumes into a whole volume file. Python modules installed into this environment in addition to parallel HDF5 should include h5py, skimage, mpi4py, glob, multiprocessing and psutil.

**\2. Python Environment for Automated Segmentation with Ilastik**

This environment is needed to segment the sub-volume files. Below is a suggestion on how to make this environment assuming the new python environment name is "lastik-devel" :

conda create -n ilastik-devel -c ilastik ilastik-everything-no-solvers

conda install -n ilastik-devel  -c conda-forge ipython

conda install -n ilastik-devel -c conda-forge  mpi4py

source activate ilastik-devel

**Note**: It is possible to run all steps of this workflow with the Ilastik/above python environment. In this case, then “making and combining sub-volumes” which takes a fraction of total processing time must be run in a single python process/thread (e.g., mpirun –np 1 python “python script”). However, the “automated segmentation with Ilastik” which takes majority of the processing time should be run by multiple python processes. 

A Command-line example to run Segmentation Script on an Image Stack With Two Python Environments
------------------------------------------------------------------------------------------------

Below is the sequence of the commands to be run on a set of networked servers to segment an image. 

**Steps to take prior to running the script**

Download all files in “v3_segment_big_data” directory. Configure the two python environments as described in the above section. If do not have “Parallel HDF5” package it is ok. The drawback is making and combining sub-volume files will take longer time (e.g., an hour vs. few minutes for terabyte sized image). 

Edit file “seg_user_param.py” to specify the sub-volume dimensions (Z, Y & X pixels), the input TIFF stack directory and the Ilastik trained file location.

**\1. Commands for “Creating Sub-volume files from TIFF Stack”**

# Activate parallel HDF python environment:

*\1. source activate parallel-phdf5*

# Convert TIFF stack into a 3D volume array asumping would like to have 4 parallel python processes

*\2. mpirun –np 4 python tiff_to_hdf5_mpi.py*

Assuming TIFF stack directory is “/home/projects/sample1_tiff” and the first file and the last file names in this directory are: sampleOne0000.tiff and sampleOne5000.tiff, then the output from the above command will be:

file “sampleOne0000_smapleOne5000.hdf5” created in a new directory called “/home/projects/sample1_tiff_mpi_hdf”

# Create sub-volume files asumping would like to have 4 parallel python processes

*\3. mpirun –np 4 python make_subvolume_mpi.py*

The above will create NNN sub-volume files in a newly created directory called “/home/projects/sample1_tiff_ilastik_inout”. For example, if volume dimension is 2000 x 1800 x 2400 and the chosen sub-volume dimension is 500 x 600 x 800 then NNN will be 036 (4x3x3) for 36 files. File names will be “sample1_tiff0000.hdf5”, "sample1_tiff0001.hdf5" to “sample1_tiff0035.hdf5”

**\2. Commands for running Automated Segmentation with Parallelized Ilastik**

# Deactivate parallel hdf5 python environment

*\1. source deactivate*

# Activate python environment to run Ilastik

*\2. source activate Ilastik-devel*

# Segment sub-volume files created in previous step assuming 12 python processes on 12 servers.

*\3. mpirun -f $HOSTLIST –np 12 python segment_subvols_pixels.py*

The above command creates NNN sub-volume files (036 files in this example) in a newly created directory called “/home/projects/sample1_tiff_pixels_maps”. Each file has K datasets (K is the number of object types/labeled classes in Ilastik trained file), each dataset is for the segmented image for a labeled class for that sub-volume.

**\3. Command for Combining Segmented Sub-volume files**

# deactivate python environment for segmentation

*\1. source deactivate*

# Activate python environment for “combining sub-volume files into volume file”

*\2. source activate parallel-phdf5*

# command to combine sub-volumes assuming would like to have 4 python processes.

*\3. mpirun –np 4 python combine_segmented_subvols.py*

The above command will create a new file called “volume_sample1_tiff_pixels_maps.h5”. This file will have K datasets for K segmented volume images.

A Command-line example to run Segmentation Script with only one python Environment
----------------------------------------------------------------------------------

Below is the sequence of commands to enter if "paralle HDF5" is not available. The same outputs as in two python environment case are created.

# Edit file “seg_user_param.py” to specify the sub-volume dimensions (Z, Y & X pixels), the input TIFF stack directory and the Ilastik trained file location.

# activate pthon environment

*\1. source activate Ilastik-devel*

# Convert TIFF stack into a 3D volume array - **must use one python processe.**

*\2. mpirun –np 1 python tiff_to_hdf5_mpi.py*

# Create sub-volume files - **must use one python processe.**

*\3. mpirun –np 1 python make_subvolume_mpi.py*

# Segment sub-volume files created in previous step assuming 12 python processes on 12 servers.

*\4. mpirun -f $HOSTLIST –np 12 python segment_subvols_pixels.py*

# combine sub-volumes into volume - **must use one python processe.**

*\5. mpirun –np 4 python combine_segmented_subvols.py*

