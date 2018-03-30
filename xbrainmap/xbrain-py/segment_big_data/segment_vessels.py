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
Module for creation of binary image with segmented vessels.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import pdb
import numpy as np
import scipy.io as sio
from scipy import ndimage as ndi
from skimage import morphology

__author__ = "Eva Dyer"
__credits__ = "Mehdi Tondravi"

__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['segment_vessels']

def segment_vessels(vessel_probability, probability_threshold, dilation_size, minimum_size):
    
    """
    This function produces a binary image with segmented vessels from a probability map (from
    ilastik or another classifier).
    
    Parameters
    ----------
    vessel_probability : ndarray
        Nr x Nc x Nz matrix which contains the probability of each voxel being a vessel.
        
    probability_threshold : float
        threshold between (0,1) to apply to probability map (only consider voxels for which
        vessel_probability(r,c,z) > probability_threshold).
        
    dilation_size : int
        Sphere Structural Element diameter size.
    
    minimum_size : int
        components smaller than this are removed from image.
    
    Returns
    -------
    ndarry
        Binary Image 
    """
    smallsize = 100 # components smaller than this size are removed.
    unfiltered_im = (vessel_probability >= probability_threshold)
    print("In segment_vessel function **** unfiltered_im.shape, dtype", unfiltered_im.shape, unfiltered_im.dtype)
    
    im_removed_small_objects = morphology.remove_small_objects(unfiltered_im, 
                                                               min_size = smallsize, in_place = True)
    dilated_im = ndi.binary_dilation(im_removed_small_objects, morphology.ball((dilation_size-1)/2))
    image_out = morphology.remove_small_objects(dilated_im, min_size = minimum_size, 
                                                in_place = True)
    return(image_out)
