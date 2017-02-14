# Hamiltonian Monte Carlo 

import copy
import sys
import numpy as np
import random
from .rateMatrix import decompHKY, decompJC, decompGTR
from .treeManipulation import idx2nodeMAP, NNI
from .Loglikelihood import phyloLoglikelihood, initialCLV 
from .Logposterior import Logpost, GradLogpost
from .refLeapfrog import reflection, refraction
from .branchManipulation import set



def hmc_iter(curr_tree, curr_branch, curr_U, D, U, beta, pden, L, nLeap, stepsz, 
             scale=0.1, delta=0.01, randomization=True, surrogate=False, include=False,
             monitor_event=False):
    # initialize the momentum
    propM = np.random.normal(size=len(curr_branch))
    currM = propM
    currH = curr_U + 0.5 * sum(currM*currM)
    propB = curr_branch
    
    # make a deep copy of the current tree
    prop_tree = copy.deepcopy(curr_tree)
    idx2node = idx2nodeMAP(prop_tree)
    
    if randomization:
        nLeap_exact = np.random.randint(1,nLeap+1)
    else:
        nLeap_exact = nLeap
    
    NNI_attempts, Ref_attempts = 0, 0
    # run the refractive leap-frog
    for i in range(nLeap_exact):
        propM = propM - stepsz/2 * GradLogpost(prop_tree, propB, D, U, beta, pden, L,
                                               scale=scale, delta=delta, surrogate=surrogate) 
        
        prop_tree, tmpB, propM, step_nni_attempts, step_ref_attempts = refraction(prop_tree, propB, propM, D, U, beta, pden, L,
                                                                                  idx2node, stepsz, delta=delta, scale=scale,
                                                                                  surrogate=surrogate, include=include)
        
        NNI_attempts += step_nni_attempts
        Ref_attempts += step_ref_attempts

        propB = tmpB     
        set(prop_tree,propB)
            
        propM = propM - stepsz/2 * GradLogpost(prop_tree, propB, D, U, beta, pden, L,
                                               scale=scale, delta=delta, surrogate=surrogate) 
    
    prop_U = Logpost(prop_tree, propB, D, U, beta, pden, L, scale)
    propH = prop_U + 0.5 * sum(propM*propM)
    
    print "NNI attempts: {}\nReflection attempts: {}".format(NNI_attempts, Ref_attempts)
    sys.stdout.flush()
    
    ratio = currH - propH
    if ratio >= min(0,np.log(np.random.uniform())):
        return prop_tree, propB, prop_U, NNI_attempts, 1, min(1.0,np.exp(ratio))
    else:
        return curr_tree, curr_branch, curr_U, NNI_attempts, 0, min(1.0,np.exp(ratio))
    

def hmc(curr_tree, curr_branch, Qmat_para, data, nLeap, stepsz, nIter, subModel='HKY', randomization=True,
        surrogate=False, burned=0.5, adap_stepsz_rate=0.4, scale=0.1, delta=0.01, include=False,
        output_filename=None):
    sampled_tree = []
    sampled_branch = []
    path_U = []
    path_Loglikelihood = []
    nni_attempts = []
    accept_rate = []
    accept_count = 0.0
    burnin = np.floor(burned*nIter)
    
    pden, sub_rate = Qmat_para
    # eigen decomposition of the rate matrix
    if subModel == 'JC':
        D, U, beta, _ = decompJC(sub_rate)
    if subModel == 'HKY':
        D, U, beta, _ = decompHKY(pden, sub_rate)
    if subModel == 'GTR':
        AG, AC, AT, GC, GT, CT = sub_rate
        D, U, beta, _ = decompGTR(pden, AG, AC, AT, GC, GT, CT)
    
    # initialize the conditional likelihood on tips
    L = initialCLV(curr_tree, data)
    
    curr_U = Logpost(curr_tree, curr_branch, D, U, beta, pden, L)
    path_U.append(curr_U)
    path_Loglikelihood.append(phyloLoglikelihood(curr_tree, curr_branch, D, U, beta, pden, L))
    
    # save current samples to files
    if output_filename:
        import uuid
        samp_tree_file = open(output_filename + '.tree','w')
        samp_para_file = open(output_filename + '.para','w')
        samp_stat_file = open(output_filename + '.stat','w')
        ID = uuid.uuid4()
        
        samp_tree_file.write('[ID:{}]\n'.format(ID))
        samp_tree_file.write('tree_0:' + '\t' + curr_tree.write(format=3) + '\n')
        samp_para_file.write('[ID:{}]\n'.format(ID))
        samp_para_file.write('nIter\t' + 'LnL\t' + '\t'.join(['length[{}]'.format(i) for i in range(len(curr_branch))]) + '\n')
        samp_para_file.write('0\t' + '{}\t'.format(path_Loglikelihood[-1]) 
                             + '\t'.join([str(branch) for branch in curr_branch]) + '\n')
        samp_stat_file.write('[ID:{}]\n'.format(ID))
        samp_stat_file.write('nIter\t' + 'AP\t' + 'NNI\n')
        samp_tree_file.flush()
        samp_para_file.flush()
        samp_stat_file.flush()
        
    Accepted = 0
    for i in range(nIter):
        exact_stepsz = np.power(1-adap_stepsz_rate,max(1-i*1.0/burnin,0))*stepsz        
        curr_tree, curr_branch, curr_U, NNI_attempts, accepted, ar = hmc_iter(curr_tree, curr_branch, curr_U,
                                                                D, U, beta, pden, L, nLeap, exact_stepsz,
                                                                scale, delta, randomization, surrogate, include)
        
        path_U.append(curr_U)
        path_Loglikelihood.append(phyloLoglikelihood(curr_tree, curr_branch, D, U, beta, pden, L))
        if output_filename:
            samp_tree_file.write('tree_{}:'.format(i+1) + '\t' + curr_tree.write(format=3) + '\n')
            samp_para_file.write('{}\t{}\t'.format(i+1,path_Loglikelihood[-1]) + 
                                                   '\t'.join([str(branch) for branch in curr_branch]) + '\n')
            samp_stat_file.write('{}\t{}\t{}\n'.format(i+1,ar,NNI_attempts))
            samp_tree_file.flush()
            samp_para_file.flush()
            samp_stat_file.flush()
            
        accept_rate.append(ar)
        nni_attempts.append(NNI_attempts)
        print "{} iteration: current Loglikelihood = {}".format(i+1,path_Loglikelihood[-1])
        sys.stdout.flush()
        if i>=burnin:
            sampled_tree.append(curr_tree)
            sampled_branch.append(curr_branch)
            accept_count += accepted
        Accepted += accepted
        if (i+1)%burnin == 0:
            print "{} iterations completed; acceptance rate: {}".format(i+1, Accepted*1.0/burnin)
            sys.stdout.flush()
            Accepted = 0
            
    if output_filename:
        samp_tree_file.close()
        samp_para_file.close()
        samp_stat_file.close()
        
    print "Overall acceptance rate: {}".format(accept_count/(nIter-burnin))
    return sampled_tree, sampled_branch, path_U, path_Loglikelihood, nni_attempts, accept_rate, accept_count 

    
   