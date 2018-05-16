import numpy as np

def KL(p, q):
    ''' 
    Compute KL divergence between two measures.
    '''
    a, b = p.ravel(), q.ravel()
    return np.sum(a * np.log(a / b))

def computeRMSE(error):
    '''
    compute the root of mean squared error.
    '''
    rmse = np.sqrt(np.mean((error)**2))
    return rmse

def computeMAE(error):
    '''
    compute the mean absolute error.
    '''
    mae = np.mean(np.abs(error))
    return mae