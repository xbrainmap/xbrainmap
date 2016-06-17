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
Module for functions related to placing an iput 3D template at a fixed position in a bounding box.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

__author__ = "Eva Dyer"
__credits__ = "Mehdi Tondravi"

__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['placeatom',
           'compute3dvec']


def placeatom(vector, box_length, which_loc, stacksz):
    
    """
    Copies the data from vector into a cube with the width of "box_length" and places the cube
    into a 3-D array with the shape/size defined by the "stacksz" parameter. The center of cube is 
    given by the "which_loc" parameter.
    
    Parameters
    ----------
    vector : ndarray
        Nx1 array
    box_length : int
        Lenght
    which_loc : int
        location to place atom in the flattened array
    stacksz : ndarry
        shape of the array (3D)
    
    Returns
    -------
    ndarray
    """
    
    output_array = np.zeros((stacksz), dtype='float32')
    
    #Convert flat index to indices 
    r, c, z = np.unravel_index(which_loc, (stacksz)) 
    output_array[r, c, z] = 1
    
    # Increase every dimension by box_length at the top and at the bottom and fill them with zeroes.
    output_array = np.lib.pad(output_array, ((box_length, box_length), (box_length, box_length), 
                           (box_length, box_length)), 'constant', constant_values=(0, 0))
    
    # get the indices of the center of cube into increased dimensions output_array.
    r, c, z = np.nonzero(output_array)
    
    #save the output of round() function to avoid multiple calls to it.
    half_length = round(box_length/2)
    
    #Save the data from the cube into output_array.
    output_array[(r - half_length +1) : (r + box_length - half_length +1), \
            (c - half_length +1) : (c + box_length - half_length +1), \
            (z - half_length +1) : (z + box_length - half_length +1)] = \
            np.reshape(vector, (box_length, box_length, box_length))
    return(output_array)


def compute3dvec(vector, which_loc, box_length, stacksz):
    
    """
    Resizes the array dimension returned by placeatom() to the shape/size given by "stacksz" parameter.
    
    Parameters
    ----------
    vector : ndarray
        Nx1 array
    box_length : int
        Lenght
    which_loc : int
        location to place atom
    stacksz : ndarry
        shape of the array (3D)
    
    Returns
    -------
    ndarray
    """
    
    output_array = placeatom(vector, box_length, which_loc, stacksz)
    
    #delete the top "box_length" arrays for all dimensions.
    x, y, z = np.shape(output_array)
    output_array = output_array[box_length:x, box_length:y, box_length:z]

    #delete the bottom "box_length" arrays for all dimensions.
    x, y, z = np.shape(output_array)
    output_array = output_array[0 : (x - box_length), 0 : (y - box_length), 0 : (z - box_length)]
    
    return output_array

