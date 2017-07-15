# Reflective, Refractive Leapfrog schemes
import copy
import numpy as np
from .Logposterior import Logpost
from .treeManipulation import idx2nodeMAP, NNI



def reflection(propB, propM, idx2node, stepsz, include=False):
    tmpB = propB + stepsz*propM
    ref_time = 0
    NNI_attempts, Ref_attempts = 0, 0
    while min(tmpB) <= 0:
        timelist = tmpB/abs(propM)
        ref_index = np.argmin(timelist)
        propB = propB + (stepsz-ref_time+timelist[ref_index]) * propM
        
        # reflect the momentum
        propM[ref_index] *= -1
        Ref_attempts += 1
        # perform NNI
        if not idx2node[ref_index].is_leaf():
            NNI_attempts += NNI(idx2node[ref_index], include=include)
        
        ref_time = stepsz + timelist[ref_index]
        tmpB = propB + (stepsz-ref_time) * propM
    
    return tmpB, propM, NNI_attempts, Ref_attempts


def refraction(prop_tree, propB, propM, D, U, U_inv, pden, L, idx2node, stepsz, 
               surrogate=True, include=False, scale = 0.1, delta = 0.01):
    tmpB = propB + stepsz*propM
    ref_time = 0
    NNI_attempts, Ref_attempts = 0, 0
    while min(tmpB) <= 0:
        timelist = tmpB/abs(propM)
        ref_index = np.argmin(timelist)
        propB = propB + (stepsz-ref_time+timelist[ref_index]) * propM
        
        propM[ref_index] *= -1
        Ref_attempts += 1
        if not idx2node[ref_index].is_leaf():
            if surrogate:
                U_before_nni = Logpost(prop_tree, propB, D, U, U_inv, pden, L, scale=scale, delta=delta, surrogate=True)
            
                tmp_tree = copy.deepcopy(prop_tree)
                tmp_idx2node = idx2nodeMAP(tmp_tree)
                # check if a topology transition has been made
                tmp_nni_made = NNI(tmp_idx2node[ref_index], include=include)
                
                if tmp_nni_made !=0:            
                    U_after_nni = Logpost(tmp_tree, propB, D, U, U_inv, pden, L, scale=scale, delta=delta, surrogate=True)
                    # compute the energy gap
                    delta_U = U_after_nni - U_before_nni
                    
                    if propM[ref_index]**2 >= 2*delta_U:
                        propM[ref_index] = np.sqrt(propM[ref_index]**2-2*delta_U)
                        prop_tree = tmp_tree
                        NNI_attempts += 1
            else:
                NNI_attempts += NNI(idx2node[ref_index], include=include)
        
        ref_time = stepsz + timelist[ref_index]
        tmpB = propB + (stepsz-ref_time) * propM
    
    return prop_tree, tmpB, propM, NNI_attempts, Ref_attempts

