
# Hamiltonian Monte Carlo 
import copy
import sys
import numpy as np
import random
from .rateMatrix import decompHKY
from .treeManipulation import idx2nodeMAP, NNI
from .Loglikelihood import phyloLoglikelihood, initialCLV 
from .Logprior import phyloLogprior_exp
from .branchManipulation import set

# a mollifier for surrogate construction
def mollifier(x, delta):
    if x >= delta:
        return x
    else:
        return 1.0/2/delta *(x*x+delta*delta)
    
# negative log-posterior
def Logpost(tree, branch, D, U, beta, pden, L, scale=0.1, surrogate=False, delta = 0.01):
    if surrogate:
        branch = [mollifier(blen,delta) for blen in branch]
        
    return -phyloLoglikelihood(tree, branch, D, U, beta, pden, L) - phyloLogprior_exp(branch, scale)


def GradLogpost(tree, branch, D, U, beta, pden, L, scale=0.1, surrogate=False, delta = 0.01, subset=None):
    rescale = 1.0
    if subset: 
        nsites = L[0].shape[1]
        L = [CV[:,subset] for CV in L]
        rescale = (nsites * rescale)/len(subset)
    
    if surrogate:
        maped_branch = [mollifier(blen,delta) for blen in branch]
        return (-phyloLoglikelihood(tree, maped_branch, D, U, beta, pden, L, grad=True) * rescale 
                - phyloLogprior_exp(maped_branch, scale, grad=True)) * np.minimum(branch,delta)/delta
    else:    
        return -phyloLoglikelihood(tree, branch, D, U, beta, pden, L, grad=True) * rescale \
               - phyloLogprior_exp(branch, scale, grad=True)
        

def reflection(propB, propM, idx2node, stepsz, include=False):
    tmpB = propB + stepsz*propM
    ref_time = 0
    NNI_attempts, Ref_attempts = 0, 0
    while min(tmpB) <= 0:
        timelist = tmpB/abs(propM)
        ref_index = np.argmin(timelist)
        propB = propB + (stepsz-ref_time+timelist[ref_index]) * propM
        
        propM[ref_index] *= -1
        Ref_attempts += 1
        # perform NNI
        if not idx2node[ref_index].is_leaf():
            NNI_attempts += NNI(idx2node[ref_index], include=include)
        
        ref_time = stepsz + timelist[ref_index]
        tmpB = propB + (stepsz-ref_time) * propM
    
    return tmpB, NNI_attempts, Ref_attempts

def refraction(prop_tree, propB, propM, D, U, beta, pden, L, idx2node, stepsz, 
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
                U_before_nni = Logpost(prop_tree, propB, D, U, beta, pden, L, scale=scale, delta=delta, surrogate=True)
            
                tmp_tree = copy.deepcopy(prop_tree)
                tmp_idx2node = idx2nodeMAP(tmp_tree)
                tmp_nni_made = NNI(tmp_idx2node[ref_index], include=include)
                
                if tmp_nni_made !=0:            
                    U_after_nni = Logpost(tmp_tree, propB, D, U, beta, pden, L, scale=scale, delta=delta, surrogate=True)
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


def hmc_iter(curr_tree, curr_branch, curr_U, D, U, beta, pden, L, nLeap, stepsz, 
             scale=0.1, delta=0.01, randomization=True, surrogate=False, include=False):
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
    

def hmc(curr_tree, curr_branch, pden, kappa, data, nLeap, stepsz, nIter, randomization=True,
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
    
    # eigen decomposition of the rate matrix
    D, U, beta, _ = decompHKY(pden, kappa)
    
    # initialize the conditional likelihood on tips
    L = initialCLV(curr_tree, data)
    
    curr_U = Logpost(curr_tree, curr_branch, D, U, beta, pden, L)
    path_U.append(curr_U)
    path_Loglikelihood.append(phyloLoglikelihood(curr_tree, curr_branch, D, U, beta, pden, L))
    
    # save current samples to files
    if output_filename:
        samp_tree_file = open(output_filename +'.tree','w')
        samp_para_file = open(output_filename + '.para','w')
        
        samp_tree_file.write('tree_0:' + '\t' + curr_tree.write(format=3) + '\n')
        samp_para_file.write('nIter\t' + 'LnL\t' + '\t'.join(['length[{}]'.format(i) for i in range(len(curr_branch))]) + '\n')
        samp_para_file.write('0\t' + '{}\t'.format(path_Loglikelihood[-1]) 
                             + '\t'.join([str(branch) for branch in curr_branch]) + '\n')
        samp_tree_file.flush()
        samp_para_file.flush()
        
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
            samp_tree_file.flush()
            samp_para_file.flush()
            
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
        
    print "Overall acceptance rate: {}".format(accept_count/(nIter-burnin))
    return sampled_tree, sampled_branch, path_U, path_Loglikelihood, nni_attempts, accept_rate, accept_count 

    
def sgld_iter(curr_tree, curr_branch, idx2node, D, U, beta, pden, L, stepsz, subsampsz=100, scale=0.1):
    subset = random.sample(range(L[0].shape[1]),subsampsz)
    propM = np.random.normal(size=len(curr_branch))

    propM = propM - stepsz/2 * GradLogpost(curr_tree, curr_branch, D, U, beta, pden, L, scale=scale, subset=subset)
    curr_branch, NNI_attempts, Ref_attempts = reflection(curr_branch, propM, idx2node, stepsz)
    
    set(curr_tree, curr_branch)
    
    return curr_branch, NNI_attempts, Ref_attempts


def sgld(init_tree, init_branch, pden, kappa, data, stepsz, nIter,
         subsampsz=100, scale=0.1, burned=0.5, printfreq=100, samplefreq=100, anneal=None):
    sampled_tree = []
    sampled_branch = []
    path_U = []
    nni_attempts = []
    burnin = np.floor(burned*nIter)
    
    # make a copy of the starting tree
    curr_tree = copy.deepcopy(init_tree)
    curr_branch = init_branch
    
    # eigen decomposition of the rate matrix
    D, U, beta, _ = decompHKY(pden, kappa)
    
    # initialize the conditional likelihood on tips
    L = initialCLV(curr_tree, data)
    
    # obtain the index to node map
    idx2node = idx2nodeMAP(curr_tree)
    
    curr_U = Logpost(curr_tree, curr_branch, D, U, beta, pden, L)
    path_U.append(curr_U)
    
    # setup annealing schedule
    if anneal:
        a, b, gamma = anneal
    else:
        a, b, gamma = stepsz, 1, 0
        
    tmp_nni_count, tmp_ref_count = 0, 0
    for i in range(nIter):
        exact_stepsz = a * np.power(b+i,-gamma)
        curr_branch, NNI_attempts, Ref_attempts = sgld_iter(curr_tree, curr_branch, idx2node, D, U, beta, pden,
                                                                       L, exact_stepsz, subsampsz, scale)
        
        tmp_nni_count += NNI_attempts
        tmp_ref_count += Ref_attempts
        if i >= burnin and (i-burnin+1)%samplefreq == 0:
            sampled_tree.append(curr_tree)
            sampled_branch.append(curr_branch)
        
        if (i+1)%printfreq == 0:
            curr_U = Logpost(curr_tree, curr_branch, D, U, beta, pden, L)
            path_U.append(curr_U)
            nni_attempts.append(tmp_nni_count)
            
            print "NNI attempts: {}\nReflection attempts: {}".format(tmp_nni_count, tmp_ref_count)
            print "{} iteration: current U = {}".format(i+1,curr_U)
            tmp_nni_count, tmp_ref_count = 0, 0
            
    
    return sampled_tree, sampled_branch, path_U, nni_attempts