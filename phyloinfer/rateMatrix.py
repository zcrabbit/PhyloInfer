# Substitution Models

import numpy as np



def decompJC(mu=1):
    pA = pG = pC = pT = .25
    beta = 4.0/(3*mu)
    rate_matrix_JC = mu/4.0 * np.ones((4,4))
    for i in range(4):
        rate_matrix_JC[i,i] = -3.0/4 * mu
    
    D_JC, U_JC = np.linalg.eig(rate_matrix_JC)
    
    return D_JC, U_JC, beta, rate_matrix_JC


def decompHKY(pden, kappa):
    pA, pG, pC, pT = pden
    beta = 1.0/(2*(pA+pG)*(pC+pT) + 2*kappa*(pA*pG+pC*pT))
    rate_matrix_HKY = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            if j!=i:
                rate_matrix_HKY[i,j] = pden[j]
            if i+j == 1 or i+j == 5:
                rate_matrix_HKY[i,j] *= kappa
    
    for i in range(4):
        rate_matrix_HKY[i,i] = - sum(rate_matrix_HKY[i,])
    
    D_HKY, U_HKY = np.linalg.eig(rate_matrix_HKY)
       
    return D_HKY, U_HKY, beta, rate_matrix_HKY


def decompGTR(pden, AG, AC, AT, GC, GT, CT):
    pA, pG, pC, pT = pden
    beta = 1.0/(2*(AG*pA*pG+AC*pA*pC+AT*pA*pT+GC*pG*pC+GT*pG*pT+CT*pC*pT))
    rate_matrix_GTR = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            if j!=i:
                rate_matrix_GTR[i,j] = pden[j]
                if i+j == 1:
                    rate_matrix_GTR[i,j] *= AG
                if i+j == 2:
                    rate_matrix_GTR[i,j] *= AC
                if i+j == 3 and abs(i-j) > 1:
                    rate_matrix_GTR[i,j] *= AT
                if i+j == 3 and abs(i-j) == 1:
                    rate_matrix_GTR[i,j] *= GC
                if i+j == 4:
                    rate_matrix_GTR[i,j] *= GT
                if i+j == 5:
                    rate_matrix_GTR[i,j] *= CT
    
    for i in range(4):
        rate_matrix_GTR[i,i] = - sum(rate_matrix_GTR[i,])
    
    D_GTR, U_GTR = np.linalg.eig(rate_matrix_GTR)
    
    return D_GTR, U_GTR, beta, rate_matrix_GTR