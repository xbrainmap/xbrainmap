=======================================================
Workflow for segmentation of Terabyte sized 3D Datasets
=======================================================

Workflow Description
--------------------
3D imaging of brain at single-neuron resolution generates Terabyte sized datasets. One of the frequently wish with such images is to segment them into biological structure of interest (e.g., vessels, cells, axon). 
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

**Note**: It is possible to run all steps of this workflow with the Ilastik/above python environment. In this case, then “making and combining sub-volumes” which takes a fraction of total processing time must be run in a single python process/thread (e.g., mpirun –np1 python “python script”). However, the “automated segmentation with Ilastik” which takes majority of the processing time should be run by multiple python processes. 

