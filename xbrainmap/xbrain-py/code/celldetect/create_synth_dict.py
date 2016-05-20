# radii coubld be 1xN vector but currently is an integer

import pdb
from strel3d import strel3d
import numpy as np
from numpy import linalg as LA
from scipy import ndimage as ndi

def create_synth_dict(radii, box_radius):
    
    Lbox = int(box_radius * 2 + 1)     #used for array dimension
    Dict = np.zeros((Lbox**3, np.size(radii)))
    cvox = int((Lbox-1)/2 + 1)
    
    for i in range(len(radii)):
        tmp = np.zeros((Lbox, Lbox, Lbox))
        tmp[cvox, cvox, cvox] = 1
        spheremat = strel3d(radii[i])
        Dict[:, i] = np.reshape(ndi.binary_dilation(tmp, spheremat), (Lbox**3))
        Dict[:, i] = Dict[:,i]/(LA.norm(Dict[:,i]))
        
    return(Dict)


'''

Below is the matlab version of this function

Dict = zeros(Lbox^3,length(radii));
cvox = (Lbox-1)/2 + 1;

for i=1:length(radii)
   tmp =zeros(Lbox,Lbox,Lbox);
   tmp(cvox,cvox,cvox)=1;
   spheremat = strel3d(radii(i));
   Dict(:,i) = reshape(imdilate(tmp,spheremat),Lbox^3,1);
   Dict(:,i) = Dict(:,i)./norm(Dict(:,i));
end

end

'''
