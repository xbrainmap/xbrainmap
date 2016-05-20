import numpy as np
from placeatom import placeatom
import pdb

def compute3dvec(vec,which_loc,Lbox,stacksz):
    
    tmp = placeatom(vec, Lbox, which_loc, stacksz)
    
    #delete the first Lbox R, C and Z 
    x,y,z = np.shape(tmp)
    tmp = tmp[Lbox:x, Lbox:y, Lbox:z]

    #delete the last Lbox R, C and Z
    x,y,z = np.shape(tmp)
    tmp = tmp[0:(x-Lbox), 0:(y-Lbox), 0:(z-Lbox)]
    return(tmp)

'''

Below is the matlab vesion of compute3dvec() function

function X = compute3dvec(vec,which_loc,Lbox,stacksz)

tmp = placeatom(vec,Lbox,which_loc,stacksz);
tmp(1:Lbox,:,:)=[];
tmp(:,1:Lbox,:)=[];
tmp(:,:,1:Lbox)=[];
tmp(end-Lbox+1:end,:,:)=[];
tmp(:,end-Lbox+1:end,:)=[];
tmp(:,:,end-Lbox+1:end)=[];
X = tmp;
 return   
end

'''
