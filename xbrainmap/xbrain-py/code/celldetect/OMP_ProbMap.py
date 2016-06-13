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
from placeatom import roundno
from convn_fft import convn_fft
from compute3dvec import compute3dvec

import numpy as np

import logging

logger = logging.getLogger(__name__)

__author__ = "Eva Dyer"
__credits__ = "Mehdi Tondravi"

__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['OMP_ProbMap']



def OMP_ProbMap(Prob, ptr, presid, startsz, dilatesz, kmax):
    
    """
    This is the top level function is to infer the position (and eventually size) of all cells in a 3D volume of 
    image data. We assume that we already have computed a "probability map" which encodes the probability that 
    each voxel corresponds to a cell body.
    
    Parameters 
    ----------
    Prob : ndarray
        Nr x Nc x Nz matrix which contains the probability of each voxel being a cell body. (i.e., 
        the (r,c,z) position of Prob contains the probability that the (r,c,z) voxel of an image cube lies 
        within a cell body.)
    ptr : float
        threshold between (0,1) to apply to probability map (only consider voxels for which Prob(r,c,z) > ptr)
    presid : float
        stopping criterion is a value between (0,1) (minimum normalized correlation between template and 
        probability map) (Example = 0.47)
    startsz : int
        initial size of spherical template (to use in sweep)
    dilatesz : int
        size to increase mask around each detected cell (zero out sphere of radius startsz+dilatesz around 
        each centroid)
    kmax : int
        maximum number of cells (alternative stopping criterion)
        
    Returns
    -------
    ndarray
        Centroids = D x 4 matrix, where D = number of detected cells.
        The (x,y,z) coordinate of each cell are contained in columns 1-3.
        The fourth column contains the correlation (ptest) between the template
        and probability map and thus represents our "confidence" in the estimate.
        The algorithm terminates when ptest<=presid.
    ndarray
        Nmap = Nr x Nc x Nz matrix containing labeled detected cells (1,...,D)
    """
    
    # threshold probability map. 
    newtest = (Prob * (Prob > ptr)).astype('float32')
    #startsz is an int now but could a vector later on - convert it to an array 
    startsz = np.atleast_1d(startsz)  
    
    # create dictionary of spherical templates
    box_radius = np.ceil(np.max(startsz)/2) + 1
    Dict = create_synth_dict(startsz, box_radius)
    Ddilate = create_synth_dict(startsz + dilatesz, box_radius)
    Lbox = roundno(np.shape(Dict)[0] ** (1/3))
    Nmap = np.zeros((np.shape(Prob)))
    newid = 1
    Centroids = np.empty((0, 4))
    
    # run greedy search step for at most kmax steps (# cells <= kmax)
    for ktot in range(kmax):
        val = np.zeros((np.shape(Dict)[1], 1))
        id = np.zeros((np.shape(Dict)[1], 1), dtype='uint32')
        # loop to convolve the probability cube with each template in Dict
        for j in range(np.shape(Dict)[1]):
            convout = convn_fft(newtest, np.reshape(Dict[:,j], (Lbox, Lbox, Lbox)))
            # get the max value of the flattened convout array and its index
            val[j],id[j] = np.real(np.amax(convout)), np.argmax(convout)
        
        # find position in image with max correlation
        which_atom = np.argmax(val)
        which_loc = id[which_atom]
        X2 = compute3dvec(Dict[:, which_atom], which_loc, Lbox,np.shape(newtest))
        xid = np.nonzero(X2)
        X3 = compute3dvec(Ddilate[:, which_atom], which_loc, Lbox, np.shape(newtest))
        
        newtest = newtest * (X3 == 0)
        ptest = val/np.sum(Dict, axis=0)
        
        if ptest < presid:
            return(Centroids, Nmap)
        
        Nmap[xid] = newid
        newid = newid + 1
        
        #Convert flat index to indices 
        rr, cc, zz = np.unravel_index(which_loc, np.shape(newtest))
        newC = cc, rr, zz  #Check - why cc is first in matlab? any connection to column-major/row-major?
        
        # insert a row into Centroids
        Centroids = np.vstack((Centroids, np.append(newC, ptest)))
        # for later: convert to logging and print with much less frequency 
        print('Iter remaining = ', (kmax - ktot - 1), 'Correlation = ', ptest )
        
    return(Centroids, Nmap)
