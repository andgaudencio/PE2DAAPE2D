import math
from numba import njit, prange
import numpy as np
from sympy.utilities.iterables import multiset_permutations


def patterns2d(X, m):
    # extract patterns from the image
    # X - image
    # m - embedding dimension
    # X_m_ - extracted patterns (vectors)

    [H, W] = np.shape(X) #H-height and W-width
    npat = (H-m+1)*(W-m+1) #number of possible patterns to extract

    X_m_ = np.empty(shape=(m*m, npat))
    X_m_[:] = 0

    cc = -1
    for row in prange(H-m+1):
        for col in prange(W-m+1):
            cc = cc + 1
            X_m_[:, cc] = np.reshape(X[row:row+m, col:col+m], (1, m*m)) #transformation of the squared pattern to a vector

    return X_m_

@njit(parallel=True)
def sort_perm(X_m_):
    #obtains the corresponding permutations patterns of X_m_ (X_m_pi) 
    [sz, den] = np.shape(X_m_) #sz = m*m, den=(H-m+1)*(W-m+1)
    X_m_pi = np.empty(shape=(sz, den))
    X_m_pi[:] = 0
    for col in prange(den):
        aux1 = np.transpose(X_m_[:, col])
        aux2 = np.argsort(aux1) #extracts the sorted positions after sorting the intensity values
        X_m_pi[:, col] = aux2

    return X_m_pi


def permute(m):
    #obtains the possible permutation patterns according to the m-value
    sz = m*m
    arr = np.zeros(shape=sz)
    for ii in range(sz):
        arr[ii] = ii

    nj = math.factorial(sz)

    pi_j = np.zeros(shape=(sz, nj))
    cc=-1
    for p in multiset_permutations(arr):
        #obtains the (m^2)! permutations
        cc = cc + 1
        pi_j[:, cc] = p

    return pi_j

@njit(parallel=True)
def get_probs_pe(pi_j, X_m_pi):
    #obtains the probability associated with each permutation pattern pi_j
    #for permutation entropy method

    sz, nj = np.shape(pi_j) #nj - number of possible pi_j
    sz, den = np.shape(X_m_pi) 

    p_p = np.zeros(shape=nj)

    for jj in prange(nj):
        for ii in prange(den):
            aux0 = (pi_j[:, jj] == X_m_pi[:, ii])
            aux1 = np.sum(aux0)
            aux2 = (aux1 == sz)
            p_p[jj] = p_p[jj] + aux2 #number of a certain pi_j

    prob_total = np.sum(p_p)
    p_p = p_p/prob_total
    return p_p

@njit(parallel=True)
def get_probs_aape(pi_j, X_m_pi, X_m_, A):
    #obtains the probability associated with each permutation pattern pi_j
    #for amplitude-aware permutation entropy method

    #pi_j - possible permutation patterns for (m*m)!
    #X_m_pi - corresponding permutation factors for the extracted patterns X_m
    #A - amplitude coefficient 

    sz, nj = np.shape(pi_j) #sz=m*m, nj    =(m*m)!
    sz, den = np.shape(X_m_pi) # den=(H-m+1)*(W-m+1)

    delta_ab = np.zeros(shape=nj)

    for jj in prange(nj):
        aux2 = 0
        for ii in prange(den):
            aux0 = (pi_j[:, ii] == X_m_pi[:, ii])
            aux1 = np.sum(aux0)

            if (aux1 == sz):
                abs_samples = np.abs(X_m_[:, ii])
                sum_samples = np.sum(abs_samples)
                aux2 = aux2 + (A/sz)*(sum_samples)
                for kk in prange(sz-1):
                    aux2 = aux2 + (1-A)/(sz)*(X_m_[kk, ii]-X_m_[kk+1, ii]) # delta_ab definition

            delta_ab[jj] = delta_ab[jj] + aux2 # sum of the corresponding delta_ab

    prob_total = np.sum(delta_ab)
    p_a = delta_ab/prob_total
    return p_a

@njit(parallel=True)
def shannon_def(prob):
    #obtains the final entropy value according to Shannon's definition
    # prob - probability values (either p_a or p_e)
    # nj - possible permutations patterns (m*m)!

    nj = len(prob)
    entropy_value = 0
    for jj in prange(nj):
        if prob[jj] > 0:
           entropy_value = entropy_value + (-prob[jj]*math.log(prob[jj]))

    return entropy_value