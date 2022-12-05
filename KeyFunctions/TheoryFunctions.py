import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.linalg import expm

def auto_corr(W, DMatrix, tau, lags):
    # Calculates the auto correlation function
    # Function inputs
    #   W: connectivity matrix
    #   tau: E and I timescale constants
    #   DMatrix: structure of input noise (i.e., independent vs. shared)
    #   lags: lags to compute the correlation over
    
    sysDim = np.shape(W)[0]
    nonpos_lags = lags[lags<=0]
    pos_lags = lags[lags>0]
    
    M = (1/tau)*(-np.eye(sysDim,sysDim) + W)
    D = (1/tau)*DMatrix
    Sigma = solve_continuous_lyapunov(-M, D@(D.T))
        
    arr1 = [Sigma @ expm((-M.T)*s) for s in nonpos_lags]
    arr2 = [expm(M*s) @ Sigma for s in pos_lags]
    
    return np.append(arr1,arr2, axis=0)
        
    
def corrTheory(W, DMatrix):
    # Computes the long-time covariance matrix
    # Notes: returns a NaN is the system is unstable
    # Function inputs
    #   W: connectivity matrix
    #   DMatrix: structure of input noise (i.e., independent vs. shared)
    # Function outputs
    #   Cov: covariance matrix
    #   Corr: correlation matrix
    #   Max_evrp: the real part of the largest eigenvalue
    #   Eigs: all eigenvalues
    
    sysDim = np.shape(W)[0]
    
    # Find the max real part of the eigenvalues
    eigs = np.linalg.eigvals(W)
    Max_evrp = np.max(np.linalg.eigvals(W).real)

    if Max_evrp >= 1:
        Cov = np.nan*np.ones(np.shape(W))
        Corr = np.nan*np.ones(np.shape(W))
    else:
        # compute covariance matrix
        M = -np.eye(sysDim) + W
        Minv = np.linalg.inv(M)
        Cov = (Minv@DMatrix) @ (Minv@DMatrix).T
        Corr = np.linalg.inv(np.sqrt(np.diag(np.diag(Cov))))@Cov@np.linalg.inv(np.sqrt(np.diag(np.diag(Cov))))
                
    return Cov, Corr, Max_evrp, eigs

def covExpansion(W, D, totalOrders):
    # Performs the expansion of the covariance matrix
    # Function inputs
    #   W: connectivity matrix
    #   DMatrix: structure of input noise (i.e., independent vs. shared)
    #   totalOrders: the number of orders wanted for the expansion
    # Function outputs
    #   covData: the contributation to the covariance for each order
    #   fullCov: the total covariance
    
    sysDim = np.shape(W)[0]
    
    innerD = D@np.transpose(D)
    
    fullCov = np.linalg.inv(np.identity(sysDim)-W)@innerD@np.transpose(np.linalg.inv(np.identity(sysDim)-W))
    
    covData = np.zeros(totalOrders)
    innerCount = 0
    for j in range(totalOrders):
        MatrixExpansion = np.zeros(np.shape(W))
        expOrder = j+1;
        # Loop over all terms
        for i in range(expOrder+1):
            MatrixExpansion = MatrixExpansion + np.linalg.matrix_power(W,i)@innerD@np.transpose(np.linalg.matrix_power(W,expOrder-i))
            covData[innerCount] = MatrixExpansion[0][1]
        innerCount = innerCount + 1
            
    return covData, fullCov
    