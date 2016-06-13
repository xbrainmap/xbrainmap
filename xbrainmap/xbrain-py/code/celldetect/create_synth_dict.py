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
Module for creatation of a collection of spherical templates of different sizes.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy import linalg as LA
from scipy import ndimage as ndi
from strel3d import strel3d

__author__ = "Eva Dyer"
__credits__ = "Mehdi Tondravi"

__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['create_synth_dict']


def create_synth_dict(radii, box_radius):
    """
    This function creates a collection of spherical templates of different sizes.
    
    Parameters
    ----------
    radii : int 
        radii coubld be 1xN vector but currently is an integer
    box_radius : float

    Returns
    -------
    ndarray
        dictionary of template vectors, of size (Lbox ** 3 x length(radii)), where Lbox = box_radius*2 +1 and 
        radii is an input to the function which contains a vector of different sphere sizes.
    """
    
    Lbox = int(box_radius * 2 + 1)     #used for array dimension
    Dict = np.zeros((Lbox**3, np.size(radii)), dtype='float32')
    cvox = int((Lbox-1)/2 + 1)
    
    for i in range(len(radii)):
        tmp = np.zeros((Lbox, Lbox, Lbox))
        tmp[cvox, cvox, cvox] = 1
        spheremat = strel3d(radii[i])
        Dict[:, i] = np.reshape(ndi.binary_dilation(tmp, spheremat), (Lbox**3))
        Dict[:, i] = Dict[:,i]/(LA.norm(Dict[:,i]))
        
    return(Dict)
