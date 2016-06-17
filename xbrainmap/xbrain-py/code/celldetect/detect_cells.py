#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
Module for detecting cells from probablity map and return their centroids.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


# following imports to be updated when directory structure are finalized 
from create_synth_dict import create_synth_dict
from compute3dvec import compute3dvec
from scipy import signal
import numpy as np

import logging

logger = logging.getLogger(__name__)

__author__ = "Eva Dyer"
__credits__ = "Mehdi Tondravi"

__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['detect_cells']


def detect_cells(cell_probability, probability_threshold, stopping_criterion, 
                initial_template_size, dilation_size, max_no_cells):
    
    """
    This is the top level function to infer the position (and eventually size) of all cells in a 3D 
    volume of image data. We assume that we already have computed a "probability map" which encodes 
    the probability that each voxel corresponds to a cell body.
    
    Parameters 
    ----------
    cell_probability : ndarray
        Nr x Nc x Nz matrix which contains the probability of each voxel being a cell body. 
    
    probability_threshold : float
        threshold between (0,1) to apply to probability map (only consider voxels for which 
        cell_probability(r,c,z) > probability_threshold)
    stopping_criterion : float
        stopping criterion is a value between (0,1) (minimum normalized correlation between 
        template and probability map) (Example = 0.47)
    initial_template_size : int
        initial size of spherical template (to use in sweep)
    dilation_size : int
        size to increase mask around each detected cell (zero out sphere of radius with 
        initial_template_size+dilation_size around each centroid)
    max_no_cells : int
        maximum number of cells (alternative stopping criterion)
        
    Returns
    -------
    ndarray
        centroids = D x 4 matrix, where D = number of detected cells.
        The (x,y,z) coordinate of each cell are in columns 1-3.
        The fourth column contains the correlation (ptest) between the template
        and probability map and thus represents our "confidence" in the estimate.
        The algorithm terminates when ptest<=stopping_criterion.
    ndarray
        new_map = Nr x Nc x Nz matrix containing labeled detected cells (1,...,D)
    """
    
    # threshold probability map. 
    newtest = (cell_probability * (cell_probability > probability_threshold)).astype('float32')
    #initial_template_size is an int now but could a vector later on - convert it to an array 
    initial_template_size = np.atleast_1d(initial_template_size)  
    
    # create dictionary of spherical templates
    box_radius = np.ceil(np.max(initial_template_size)/2) + 1
    dict = create_synth_dict(initial_template_size, box_radius)
    dilate_dict = create_synth_dict(initial_template_size + dilation_size, box_radius)
    box_length = round(np.shape(dict)[0] ** (1/3))
    new_map = np.zeros((np.shape(cell_probability)), dtype='uint8')
    newid = 1
    centroids = np.empty((0, 4))
    
    # run greedy search step for at most max_no_cells steps (# cells <= max_no_cells)
    for ktot in range(max_no_cells):
        val = np.zeros((np.shape(dict)[1], 1), dtype='float32')
        id = np.zeros((np.shape(dict)[1], 1), dtype='uint32')
        
        # loop to convolve the probability cube with each template in dict
        for j in range(np.shape(dict)[1]):
            convout = signal.fftconvolve(newtest, np.reshape(dict[:,j], (box_length, box_length, 
                                                                         box_length)), mode='same')
            # get the max value of the flattened convout array and its index
            val[j],id[j] = np.real(np.amax(convout)), np.argmax(convout)
        
        # find position in image with max correlation
        which_atom = np.argmax(val)
        which_loc = id[which_atom]
        
        # Save dict into a cube array with its center given by which_loc and place it into a 3-D array.
        x2 = compute3dvec(dict[:, which_atom], which_loc, box_length, np.shape(newtest))
        xid = np.nonzero(x2)
        
        # Save dilate_dict into a cube array with its center given by which_loc and place it into a 3-D array. 
        x3 = compute3dvec(dilate_dict[:, which_atom], which_loc, box_length, np.shape(newtest))
        
        newtest = newtest * (x3 == 0)
        ptest = val/np.sum(dict, axis=0)
        
        if ptest < stopping_criterion:
            return(centroids, new_map)
        
        # Label detected cell
        new_map[xid] = newid
        newid = newid + 1
        
        #Convert flat index to indices 
        rr, cc, zz = np.unravel_index(which_loc, np.shape(newtest))
        new_centroid = cc, rr, zz  #Check - why cc is first?
        
        # insert a row into centroids
        centroids = np.vstack((centroids, np.append(new_centroid, ptest)))
        # for later: convert to logging and print with much less frequency 
        print('Iter remaining = ', (max_no_cells - ktot - 1), 'Correlation = ', ptest )
        
    return(centroids, new_map)
