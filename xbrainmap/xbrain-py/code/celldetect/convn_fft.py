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
Module for computing  n-dimensional convolution in the Fourier domain
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy.fftpack import fftn, ifftn


__author__ = "Eva Dyer"
__credits__ = "Mehdi Tondravi"

__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['nextpow2',
           'convn_fft']


def nextpow2(number):
    
    """
    nextpow2(I) returns the exponent for the smallest powers of two that satisfy 2 ** n >= number.
    It can be used to pad an array with zeros to the next power of 2 for faster fft computation.
    
    Parameters
    ----------
    number : int
        
    Returns
    -------
    int
    """
    n = 0
    while 2**n < number: n += 2
    return n


def convn_fft(a,b):
    
    """
    This computes a n-dimensional convolution in the Fourier domain (uses fft rather than spatial
    convolution to reduce complexity).
    
    Parameters
    ----------
    a : ndarray
    b : ndarray

    Returns
    -------
    
    """
    sz_a = np.array(a.shape)
    sz_b = np.array(b.shape)
    sz_a_plus_b = sz_a + sz_b
    pow2 = np.array([], dtype='uint32')

    # pad with zeros to the next power of 2 to speed up fft
    for i in sz_a_plus_b:
        pow2 = np.append(pow2, [nextpow2(i)])
    
    idx = [[]] * 3
    c = fftn(a, 2**pow2) 
    c = c*(fftn(b, 2**pow2))
    c = ifftn(c)
    
    for i in range(3):
        idx[i] = int(np.ceil((sz_b[i] - 1) / 2)) + np.arange(0, (sz_a[i] + 1))
    
    return c[idx[0][0]:idx[0][-1], idx[1][0]:idx[1][-1], idx[2][0]:idx[2][-1]]
