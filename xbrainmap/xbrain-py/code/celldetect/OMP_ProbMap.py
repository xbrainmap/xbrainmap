'''

%%%%%%%%%%%%%%%%%%%%%%
% OMP_ProbMap.m = function to detect cells and return their centroids
%%%
% Input
%%%
% Prob = Nr x Nc x Nz matrix which contains the probability of each voxel being a cell body. (i.e., the (r,c,z) position of Prob contains the probability that the (r,c,z) voxel of an image cube lies within a cell body.)
% ptr = threshold between (0,1) to apply to probability map (only consider voxels for which Prob(r,c,z) > ptr)
% presid = stopping criterion is a value between (0,1) (minimum normalized correlation between template and probability map) (Example = 0.47)
% startsz = initial size of spherical template (to use in sweep)
% dilatesz = size to increase mask around each detected cell (zero out sphere of radius startsz+dilatesz around each centroid)
% kmax = maximum number of cells (alternative stopping criterion)
%%%
% Output
%%%
% Centroids = D x 4 matrix, where D = number of detected cells. 
%             The (x,y,z) coordinate of each cell are contained in columns 1-3. 
%             The fourth column contains the correlation (ptest) between the template 
%             and probability map and thus represents our "confidence" in the estimate. 
%             The algorithm terminates when ptest<=presid.
% Nmap = Nr x Nc x Nz matrix containing labeled detected cells (1,...,D)
%%%%%%%%%%%%%%%%%%%%%%

'''

import pdb
from create_synth_dict import create_synth_dict
from placeatom import roundno
from convn_fft import convn_fft
from compute3dvec import compute3dvec
import numpy as np

def OMP_ProbMap(Prob, ptr, presid, startsz, dilatesz, kmax):
    
    # threshold probability map. 
    #Will check later - Is copying Prob needed? No copy modifies Prob in caller function. In test data Prob is 512MB
    newtest = np.copy(Prob * (Prob > ptr))
    #startsz could be scalar or a vector - convert it to an array if scalar
    startsz = np.atleast_1d(startsz)  
    
    # create dictionary of spherical templates
    box_radius = np.ceil(np.max(startsz)/2) + 1
    Dict = create_synth_dict(startsz, box_radius)
    Ddilate = create_synth_dict(startsz + dilatesz, box_radius)
    Lbox = roundno(np.size(Dict)**(1/3))
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
        
        newid = newid + 1
        newtest = newtest * (X3 == 0)
        ptest = val/np.sum(Dict, axis=0)
        
        if ptest < presid:
            return
        
        Nmap[xid] = newid
        newid = newid + 1
        
        #Convert flat index to indices 
        rr, cc, zz = np.unravel_index(which_loc, np.shape(newtest))
        newC = rr, cc, zz  #Check - why cc is first in matlab? any connection to column-major/row-major?
        
        # insert a row into Centroids
        Centroids = np.vstack((Centroids, np.append(newC, ptest)))
        
        print('Iter remaining = ', (kmax - ktot - 1), 'Correlation = ', ptest )
        pdb.set_trace()
        
    return


'''

Below is the matlab version of this function

function [Centroids,Nmap] = OMP_ProbMap(Prob,ptr,presid,startsz,dilatesz,kmax)

% threshold probability map
Prob = Prob.*(Prob>ptr);

% create dictionary of spherical templates
box_radius = ceil(max(startsz)/2) + 1;
Dict = create_synth_dict(startsz,box_radius);
Ddilate = create_synth_dict(startsz+dilatesz,box_radius);
Lbox = round(length(Dict)^(1/3));

Nmap = zeros(size(Prob));
newid = 1;
newtest = Prob;
Centroids = [];

% run greedy search step for at most kmax steps (# cells <= kmax)
for ktot = 1:kmax
    tic,
    val = zeros(size(Dict,2),1);
    id = zeros(size(Dict,2),1);
    
    for j = 1:size(Dict,2)
       convout = convn_fft(newtest,reshape(Dict(:,j),Lbox,Lbox,Lbox));
       [val(j),id(j)] = max(convout(:)); % positive coefficients only
    end
    
    % find position in image with max correlation
    [~,which_atom] = max(val); 
    which_loc = id(which_atom); 
  
    X2 = compute3dvec(Dict(:,which_atom),which_loc,Lbox,size(newtest));
    xid = find(X2); 

    X3 = compute3dvec(Ddilate,which_loc,Lbox,size(newtest));
    
    newid = newid+1;
    newtest = newtest.*(X3==0);
    ptest = val./sum(Dict);
    
    if ptest<presid
        return
    end
    Nmap(xid) = newid;
    newid = newid+1;

    [rr,cc,zz] = ind2sub(size(newtest),which_loc);
    newC = [cc, rr, zz];

    Centroids = [Centroids; [newC,ptest]];

    display(['Iter remaining = ', int2str(kmax-ktot), ...
         ' Correlation = ', num2str(ptest,3)])

end

end

'''
