from scipy import signal
from helper_functions.helpers import * 
import numpy as np
from scipy.linalg import expm

def xcov(x,y):
    # computes the (bias-adjusted) cross-covariance of x and y
    xbar = np.mean(x)
    ybar = np.mean(y)
    n = len(x)
    d = np.arange(n)
    k = np.append(d[::-1], d[1:])
    return signal.correlate(x-xbar, y-ybar, mode='full', method='fft')/(n-k)

def stationary_acovf(lags, M, sigma):
    nonpos_lags = lags[lags<=0]
    pos_lags = lags[lags>0]
    arr1 = [sigma @ expm((-M.T)*s) for s in nonpos_lags]
    arr2 = [expm(M*s) @ sigma for s in pos_lags]
    return np.append(arr1,arr2, axis=0)

def autocov_sim(T=2500, tau_E=15, tau_I=15, sigE=1, sigI=1, c=0.5, shared_structure=np.array([[1],[1],[0]]), Wee=0.5, Wii=0.5, Wei=0.5, Wie=0.5, mu_0=np.array([[30.25], [30.25], [22.5]]), alpha=0.15, num_of_sims=10):
    
    for sim in range(num_of_sims):
        print(f'{sim} of {num_of_sims-1}')
        if sim == 0: # in the first sim, initialize data matrices
            r_ss, r_n, r_p, R_ss, R_n, R_p, W, muu, ts, s_e, s_i = global_inh_model(t0=0,
                                                                                    r0=0,
                                                                                     T=T, 
                                                                                     dt=.01,
                                                                                     t_kick=80_000,
                                                                                     kick_bool=False,
                                                                                     kick=.5,
                                                                                     tau_E=tau_E, 
                                                                                     tau_I=tau_I, 
                                                                                     sigE=sigE, 
                                                                                     sigI=sigI,
                                                                                     c=c,
                                                                                     shared_structure=shared_structure,
                                                                                     Wee=Wee,
                                                                                     Wii=Wii,
                                                                                     Wei=Wei,
                                                                                     Wie=Wie,
                                                                                     mu=mu_0,
                                                                                     alpha = alpha)


            R_n = R_n[:, 20_000:] # clips transient
            R_ss = R_ss[:, 20_000:]  

            corrmat11 = xcov(R_n[0,:], R_n[0,:]) 
            corrmat22 = xcov(R_n[1,:], R_n[1,:])
            corrmat23 = xcov(R_n[1,:], R_n[2,:])
            corrmat12 = xcov(R_n[0,:], R_n[1,:])
            corrmat13 = xcov(R_n[0,:], R_n[2,:])
            corrmat33 = xcov(R_n[2,:], R_n[2,:])
        
        else: # in other sims, append each sim as a column to current data matrix
            r_ss, r_n, r_p, R_ss, R_n, R_p, W, muu, ts, s_e, s_i = global_inh_model(t0=0,
                                                                                    r0=0,
                                                                                     T=T, 
                                                                                     dt=.01,
                                                                                     t_kick=80_000,
                                                                                     kick_bool=False,
                                                                                     kick=.5,
                                                                                     tau_E=tau_E, 
                                                                                     tau_I=tau_I, 
                                                                                     sigE=sigE, 
                                                                                     sigI=sigI,
                                                                                     c=c,
                                                                                     shared_structure=shared_structure,
                                                                                     Wee=Wee,
                                                                                     Wii=Wii,
                                                                                     Wei=Wei,
                                                                                     Wie=Wie,
                                                                                     mu=mu_0,
                                                                                     alpha = alpha)


            R_n = R_n[:, 20_000:] # clips transient
            R_ss = R_ss[:, 20_000:]
            
            corrmat11 = np.c_[corrmat11, xcov(R_n[0,:], R_n[0,:])] 
            corrmat22 = np.c_[corrmat22, xcov(R_n[1,:], R_n[1,:])]
            corrmat23 = np.c_[corrmat23, xcov(R_n[1,:], R_n[2,:])]
            corrmat12 = np.c_[corrmat12, xcov(R_n[0,:], R_n[1,:])]
            corrmat13 = np.c_[corrmat13, xcov(R_n[0,:], R_n[2,:])]
            corrmat33 = np.c_[corrmat33, xcov(R_n[2,:], R_n[2,:])]
            
        
    return corrmat11, corrmat22, corrmat23, corrmat12, corrmat13, corrmat33
    
    
    
            
            
            
            
            
            
             