
# Logprior and its gradient
import numpy as np


# exponential prior for the branch lengths
def phyloLogprior_exp(branch, scale=0.1, grad=False):
    if not grad:
        return -np.sum(branch)/scale - np.log(scale)*len(branch)
    else:
        return -np.ones(len(branch))/scale