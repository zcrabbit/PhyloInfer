# Compute the log-posterior function

from .Loglikelihood import phyloLoglikelihood
from .Logprior import phyloLogprior_exp
import numpy as np



# a mollifier for surrogate construction
def mollifier(x, delta):
    if x >= delta:
        return x
    else:
        return 1.0/2/delta *(x*x+delta*delta)
    
# negative log-posterior
def Logpost(tree, branch, D, U, U_inv, pden, L, scale=0.1, surrogate=False, delta = 0.01):
    if surrogate:
        branch = [mollifier(blen,delta) for blen in branch]
        
    return -phyloLoglikelihood(tree, branch, D, U, U_inv, pden, L) - phyloLogprior_exp(branch, scale)


def GradLogpost(tree, branch, D, U, pden, L, scale=0.1, surrogate=False, delta = 0.01):
    if surrogate:
        maped_branch = [mollifier(blen,delta) for blen in branch]
        return (-phyloLoglikelihood(tree, maped_branch, D, U, U_inv, pden, L, grad=True)  
                - phyloLogprior_exp(maped_branch, scale, grad=True)) * np.minimum(branch,delta)/delta
    else:    
        return -phyloLoglikelihood(tree, branch, D, U, U_inv, pden, L, grad=True) - phyloLogprior_exp(branch, scale, grad=True)