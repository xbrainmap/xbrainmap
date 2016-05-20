import numpy as np
import pdb

def roundno(no):
    
    '''
    python rounds to the nearest even value. For exampl (12.5) returns 12. But in matlab
    round(12.5) returns 13. This function is to have the same behavior as in matlab code. Inconsistency
    is only if the number ends with ".5".
    '''
    
    return int(no // 1 + ((no % 1) / 0.5) // 1)

def placeatom(vec,Lbox, which_loc, stacksz):
    
    tmp = np.zeros((stacksz))
    #Convert flat index to indices 
    r,c,z = np.unravel_index(which_loc, (stacksz)) 
    tmp[r, c, z] = 1
    
    # Increase every dimension by Lbox before, Lbox after each dimension and fill them with zeros
    tmp = np.lib.pad(tmp, ((Lbox, Lbox), (Lbox, Lbox), (Lbox, Lbox)), 'constant', constant_values=(0, 0))
    # get the indices of the nonzero element 
    center_loc = np.nonzero(tmp)
    Lbox_half = roundno(Lbox / 2)
    
    tmp[center_loc[0] - Lbox_half + 1:center_loc[0] + Lbox_half, \
            center_loc[1] - Lbox_half + 1:center_loc[1] + Lbox_half, \
            center_loc[2] - Lbox_half + 1:center_loc[2] + Lbox_half] = \
            np.reshape(vec, (Lbox, Lbox, Lbox))
    return(tmp)


'''

Below is the matlab vesion of placeatom() function

function X = placeatom(vec,Lbox,which_loc,stacksz)

tmp = zeros(stacksz);
tmp(which_loc)=1;
tmp = padarray(tmp,[Lbox,Lbox,Lbox]);
whichloc2 = find(tmp);
[center_loc(1),center_loc(2),center_loc(3)] = ind2sub(size(tmp),whichloc2);
        tmp(center_loc(1)-round(Lbox/2)+1:center_loc(1)+round(Lbox/2)-1,...
        center_loc(2)-round(Lbox/2)+1:center_loc(2)+round(Lbox/2)-1,...
        center_loc(3)-round(Lbox/2)+1:center_loc(3)+round(Lbox/2)-1) = ...
        reshape(vec,Lbox,Lbox,Lbox);    
X = tmp;
end

'''
