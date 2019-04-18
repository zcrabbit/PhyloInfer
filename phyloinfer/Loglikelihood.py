
# Loglikelihood and its gradients
import numpy as np
# from warnings import warn
import warnings
warnings.simplefilter('always', UserWarning)
nuc2vec = {'A':[1.,0.,0.,0.], 'G':[0.,1.,0.,0.], 'C':[0.,0.,1.,0.], 'T':[0.,0.,0.,1.],
           '-':[1.,1.,1.,1.], '?':[1.,1.,1.,1.], 'N':[1.,1.,1.,1.], 'R':[1.,1.,0.,0.],
           'Y':[0.,0.,1.,1.], 'S':[0.,1.,1.,0.], 'W':[1.,0.,0.,1.], 'K':[0.,1.,0.,1.],
           'M':[1.,0.,1.,0.], 'B':[0.,1.,1.,1.], 'D':[1.,1.,0.,1.], 'H':[1.,0.,1.,1.],
           'V':[1.,1.,1.,0.], '.':[1.,1.,1.,1.], 'U':[0.,0.,0.,1.]}


# initialize the conditional likelihood vectors, added unique site option
def initialCLV(data, unique_site=False):
    nseqs, nsites = len(data), len(data[0])
    if unique_site:
        data_arr = np.array(zip(*data))
        unique_sites, counts = np.unique(data_arr, return_counts=True, axis=0)
        n_unique_sites = len(counts)
        unique_data = unique_sites.T
        
        L = np.ones((2*nseqs-2, 4, n_unique_sites))
        for i in range(nseqs):
            L[i] = np.transpose([nuc2vec[c] for c in unique_data[i]])
        return L, counts
    else:   
        L = np.ones((2*nseqs-2, 4, nsites))
        for i in range(nseqs):
            L[i] = np.transpose([nuc2vec[c] for c in data[i]])
        return L

            
def phyloLoglikelihood(tree, branch, D, U, U_inv, pden, L, site_counts=1.0, grad=False, value_and_grad=False):
    nsites = L[0].shape[1]

    Loglikelihood = 0 
    GradLoglikelihood = np.zeros(len(branch))    
    Down = [1.0] * len(branch)
    pt_matrix = [0.0] * len(branch)
    
    for node in tree.traverse("postorder"):
        if not node.is_leaf():
            L[node.name] = 1.0
            for child in node.children:
                pt_matrix[child.name] = np.dot(U * np.exp(D*branch[child.name]), U_inv)
                Down[child.name] = np.dot(pt_matrix[child.name],L[child.name])
                L[node.name] *= Down[child.name] 
            scaler = np.sum(L[node.name],axis=0)
            if not np.all(scaler>0):
                warnings.warn("Caution! Incompatible data detected!")
                return -np.inf
                
            L[node.name] /= scaler
            Loglikelihood += np.sum(np.log(scaler) * site_counts)
            
    Loglikelihood += np.sum(np.log(np.dot(pden,L[node.name])) * site_counts)
    
    if not grad:
        return Loglikelihood
    
    Up = [1.0] * (len(branch)+1)
    for node in tree.traverse("preorder"):
        if node.is_root():
            Up[node.name] = np.repeat(pden.reshape(-1,1), nsites, axis=1)
        else:
            for sister in node.get_sisters():
                Up[node.name] *= Down[sister.name]
            Up[node.name] *= Up[node.up.name]
            pt_matrix_grad = np.dot(U * (D*np.exp(D*branch[node.name])), U_inv)
            
            gradient = np.sum(Up[node.name] * np.dot(pt_matrix_grad,L[node.name]),axis=0)
            
            Up[node.name] = np.dot(pt_matrix[node.name].T,Up[node.name])
            gradient /= np.sum(Up[node.name]* L[node.name],axis=0)
            GradLoglikelihood[node.name] = np.sum(gradient * site_counts)
            
            if not node.is_leaf():
                scaler = np.sum(Up[node.name],axis=0)
                Up[node.name] /= scaler  
    
    if not value_and_grad:            
        return GradLoglikelihood
    else:
        return Loglikelihood, GradLoglikelihood


# compute the log marginal likelihood of a tree given conjugate Gamma prior for branch lengths
def phyloLogMarginallikelihood(tree, D, U, pden, L, gamma_prior=(1,10)):
    nsites = L[0].shape[1]
    ntips = len(tree)
    marginalLoglikelihood = 0    
    Down = [np.ones((4,nsites)) for i in range(2*ntips-3)]
    marginal_pt_matrix = [np.zeros((4,4)) for i in range(2*ntips-3)]
    k, theta = gamma_prior
    
    for node in tree.traverse("postorder"):
        if not node.is_leaf():
            L[node.name] = np.ones((4,nsites))
            for child in node.children:
                marginal_pt_matrix[child.name] = np.transpose(np.linalg.lstsq(U.T, 
                                np.dot(np.diag(np.power(theta/(theta-D),k)),U.T))[0])
                Down[child.name] = np.dot(marginal_pt_matrix[child.name],L[child.name])
                L[node.name] *= Down[child.name] 
            scaler = np.amax(L[node.name],axis=0)
            L[node.name] /= scaler
            marginalLoglikelihood += np.sum(np.log(scaler))
            
    marginalLoglikelihood += np.sum(np.log(np.dot(pden,L[node.name])))

    return marginalLoglikelihood