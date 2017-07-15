
# Loglikelihood and its gradients
import numpy as np
from warnings import warn
nuc2vec = {'A':[1.,0.,0.,0.], 'G':[0.,1.,0.,0.], 'C':[0.,0.,1.,0.], 'T':[0.,0.,0.,1.],
           '-':[1.,1.,1.,1.], '?':[1.,1.,1.,1.]}



# initialized the conditional likelihood vectors
def initialCLV(data):
    nseqs, nsites = len(data), len(data[0])
    
    L = [np.ones((4,nsites)) for i in range(2*nseqs-2)]
    for i in range(nseqs):
        L[i] = np.transpose([nuc2vec[c] for c in data[i]])
    return L

            
def phyloLoglikelihood(tree, branch, D, U, U_inv, beta, pden, L, grad=False):
    nsites = L[0].shape[1]
    ntips = len(tree)
    Loglikelihood = 0 
    GradLoglikelihood = np.zeros(2*ntips-3)    
    Down = [np.ones((4,nsites)) for i in range(2*ntips-3)]
    pt_matrix = [np.zeros((4,4)) for i in range(2*ntips-3)]
    
    for node in tree.traverse("postorder"):
        if not node.is_leaf():
            L[node.name] = np.ones((4,nsites))
            for child in node.children:
                # pt_matrix[child.name] = np.transpose(np.linalg.lstsq(U.T,
                #                 np.dot(np.diag(np.exp(D*branch[child.name]*beta)),U.T))[0])
                pt_matrix[child.name] = np.dot(U, np.dot(np.diag(np.exp(D*branch[child.name])), U_inv))
                Down[child.name] = np.dot(pt_matrix[child.name],L[child.name])
                L[node.name] *= Down[child.name] 
            scaler = np.amax(L[node.name],axis=0)
            if not all(scaler):
                warn("Caution! Incompatible data detected!")
                return -np.inf
                
            L[node.name] /= scaler
            Loglikelihood += np.sum(np.log(scaler))
            
    Loglikelihood += np.sum(np.log(np.dot(pden,L[node.name])))
    
    if not grad:
        return Loglikelihood
    
    Up = [np.ones((4,nsites)) for i in range(2*ntips-2)]
    for node in tree.traverse("preorder"):
        if node.is_root():
            Up[node.name] = (pden * Up[node.name].T).T
        else:
            for sister in node.get_sisters():
                Up[node.name] *= Down[sister.name]
            Up[node.name] *= Up[node.up.name]
            # pt_matrix_grad = np.transpose(np.linalg.lstsq(U.T,
            #                     np.dot(np.diag(beta*D*np.exp(D*branch[node.name]*beta)),U.T))[0])
            pt_matrix_grad = np.dot(U, np.dot(np.diag(D*np.exp(D*branch[node.name])), U_inv))
            
            gradient = np.sum(Up[node.name] * np.dot(pt_matrix_grad,L[node.name]),axis=0)
            
            Up[node.name] = np.dot(pt_matrix[node.name].T,Up[node.name])
            gradient /= np.sum(Up[node.name]* L[node.name],axis=0)
            GradLoglikelihood[node.name] = np.sum(gradient)
            
            if not node.is_leaf():
                scaler = np.amax(Up[node.name],axis=0)
                Up[node.name] /= scaler  
                
    return GradLoglikelihood


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