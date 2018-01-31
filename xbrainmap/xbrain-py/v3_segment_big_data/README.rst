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

