import numpy as np
from scipy.optimize import linprog


#########################################
## Compute Optimal Transport Distance  ##
#########################################
def ot(C, r, c):
    '''
    This function comutes the optimal transport distance between probability measures
    r and c under cost matrix C.
    
    Parameters:
        C: cost matrix, (m, n) np.ndarray
        r: row marginal, np.array
        c: column marginal, np.array
    
    Return:
        opt_pi: (m, n) matrix
        opt_cost: scalar
    '''
    m, n =  C.shape

    cost = C.reshape(m * n)
    
    A = np.zeros([m + n, m * n])

    for i in range(m):
        A[i][i * n : (i + 1) * n] = 1
    for i in range(m, m + n):
        A[i][i - m: m * n : n] = 1
    
    b = np.append(r, c)

    bounds = [(0, None)] * (m * n)

    res = linprog(cost, A_eq=A, b_eq=b, bounds=bounds)

    opt_pi. opt_cost = res.x.reshape((m, n)), res.fun
    
    return opt_pi, opt_cost


#############################################################################
#### Compute Sinkhorn Distance (Regularized Optimal Transport Distance) #####
#############################################################################
def rot(C, r, c, lam=1.0, max_iteration=100, tol=1e-6):
    '''
    This function computes the Sinkhorn distance between two discrete
    distributions r and c under cost matrix C

    Parameters:
        C: cost matrix, (m, n) np.ndarray
        r: row marginal, np.array
        c: column marginal, np.array
        lam: regularization constant
        L: number of iteartions in Sinkhorn-Knopp algorithm

    Return:
        pi: regularized optimal transport plan, (m, n) np.ndarray
        a: left scaling factor, np.array
        b: right scaling factor, np.array
        opt_cost: optimal sinkhorn distance, scalar
    '''
    m, n  = C.shape

    K = np.exp(-lam * C)

    a, b = np.ones(m), np.ones(n)

    iteration, error = 0, np.inf

    while iteration < max_iteration and error > tol:
        next_b = c / np.dot(K.T, a)
        next_a = r / np.dot(K, next_b)
        error = np.linalg.norm(next_a - a) + np.linalg.norm(next_b - b)
        a, b = next_a, next_b
    
    pi = np.dot(np.diag(a),  K * b)
    
    loss = np.sum(pi * C) + 1 / lam * (np.sum(pi * np.log(pi)) - 1)

    return pi, a, b, loss







    
    