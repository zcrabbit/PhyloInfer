# Substitution Models

import numpy as np



def decompJC():
    pA = pG = pC = pT = .25
    rate_matrix_JC = 1/3.0 * np.ones((4,4))
    for i in range(4):
        rate_matrix_JC[i,i] = -1.0
    
    D_JC, U_JC = np.linalg.eig(rate_matrix_JC)
    U_JC_inv = np.linalg.inv(U_JC)
    
    return D_JC, U_JC, U_JC_inv, rate_matrix_JC


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
    
    rate_matrix_HKY = beta * rate_matrix_HKY
    D_HKY, U_HKY = np.linalg.eig(rate_matrix_HKY)
    U_HKY_inv = np.linalg.inv(U_HKY)
       
    return D_HKY, U_HKY, U_HKY_inv, rate_matrix_HKY


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
    
    rate_matrix_GTR = beta * rate_matrix_GTR
    D_GTR, U_GTR = np.linalg.eig(rate_matrix_GTR)
    U_GTR_inv = np.linalg.inv(U_GTR)
    
    return D_GTR, U_GTR, U_GTR_inv, rate_matrix_GTR