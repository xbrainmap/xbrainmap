'''

% function se=STREL3D(sesize)
%
% STREL3D creates a 3D sphere as a structuring element. Three-dimensional 
% structuring elements are much better for morphological reconstruction and
% operations of 3D datasets. Otherwise the traditional MATLAB "strel"
% function will only operate on a slice-by-slice approach. This function
% uses the aribtrary neighborhood for "strel."
% 
% Usage:        se=STREL3D(sesize)
%
% Arguments:    sesize - desired diameter size of a sphere (any positive 
%               integer)
%
% Returns:      the structuring element as a strel class (can be used
%               directly for imopen, imclose, imerode, etc)
% 
% Examples:     se=strel3d(1)
%               se=strel3d(2)
%               se=strel3d(5)
%
% 2014/09/26 - LX 
% 2014/09/27 - simplification by Jan Simon
'''
'''

Below function (ball) is from skimage.morphology with only one line changed;

Changed from:
n = 2 * radius + 1

To:
n = 2 * radius

With this change Structural Element (SE) ball dimensions are the same as the SE generated 
by Matlabcode.

'''

import numpy as np

def ball(radius, dtype=np.uint8):
    
    n = 2 * radius
    Z, Y, X = np.mgrid[-radius:radius:n * 1j,
                       -radius:radius:n * 1j,
                       -radius:radius:n * 1j]
    s = X ** 2 + Y ** 2 + Z ** 2
    return np.array(s <= radius * radius, dtype=dtype)


#from skimage.morphology import ball

def strel3d(sesize):
    strel_sphere = ball(int(sesize/2))
    return(strel_sphere)


'''

Below is the matlab version of function strel3d() in xbrain/code/utils/filtercc directory

function se=strel3d(sesize)

sw=(sesize-1)/2; 
ses2=ceil(sesize/2);            % ceil sesize to handle odd diameters
[y,x,z]=meshgrid(-sw:sw,-sw:sw,-sw:sw); 
m=sqrt(x.^2 + y.^2 + z.^2); 
b=(m <= m(ses2,ses2,sesize)); 
se=strel('arbitrary',b);

'''

