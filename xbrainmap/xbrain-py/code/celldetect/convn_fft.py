def nextpow2(number):
    
    '''
    nextpow2(I) returns the exponent for the smallest powers of two that satisfy 2 ** n >= number.
    It can be used to pad an array with zeros to the next power of 2 for faster fft computation.
    
    '''
    n = 0
    while 2**n < number: n += 2
    return n

import numpy as np
from scipy.fftpack import fftn, ifftn
import pdb

def convn_fft(a,b):
    
    sz_a = np.array(a.shape)
    sz_b = np.array(b.shape)
    sz_a_plus_b = sz_a + sz_b
    pow2 = np.array([], dtype='uint32')

    # pad pow2 with zeros to the next power of 2 to speed up fft
    for i in sz_a_plus_b:
        pow2 = np.append(pow2, [nextpow2(i)])
    
    # need to figure out how to convert to single percision same as matlab version of the code
    c = fftn(a, 2**pow2) 
    c = c*(fftn(b, 2**pow2))
    c = ifftn(c)
    
    idx = [[]] * 3
    for i in range(3):
        idx[i] = int(np.ceil((sz_b[i] - 1) / 2)) + np.arange(0, (sz_a[i] + 1))
    
    return c[idx[0][0]:idx[0][-1], idx[1][0]:idx[1][-1], idx[2][0]:idx[2][-1]]
    

'''

Below is the matlab vesion of convn_fft() function

function out = convn_fft(a,b)
% A is larger input
% B is template

sz_a = size(a);
sz_b = size(b);
pow2 = nextpow2(sz_a+sz_b);

c = single(fftn(a,2.^pow2));
c = c.*single(fftn(b,2.^pow2));
c = ifftn(c);

if 0 %original
A = fftn(a,2.^pow2);
B = fftn(b,2.^pow2);
C = A.*B;
c2 = ifftn(C);
end

for i=1:3;
    idx{i} = ceil((sz_b(i)-1)/2)+(1:sz_a(i));
end

out = c(idx{1},idx{2},idx{3});
    

end

'''
