import numpy as np
from scipy import signal

def global_inh_model(r0, r_bar, T, dt, W, tau, DMatrix):
    # Simulates the rate equations
    # Function inputs
    #   r0: rates to start at
    #   r_bar: desired steady state solution
    #   T: sim end 
    #   dt: size of timestep 
    #   W: connectivity matrix
    #   tau: E and I timescale constants
    #   DMatrix: structure of input noise (i.e., independent vs. shared)
    
    # Total number of timesteps
    M = int(T/dt)
    
    # Dimension of W
    sysDim = np.shape(W)[0]
    
    # Preallocate memory: matrices storing firing rates as row and time as column
    R_n = np.zeros((sysDim,M+1))
                
    mu = (np.eye(sysDim)-W)@r_bar # stimulus inputs required for given r_bar    
    # Initial condition for the system
    r_n = r0
    R_n[:,[0]] = r0
    
    # white noise process: first sysDim are private noise terms, last is shared noise term
    x = np.zeros((sysDim+1,1)) 
    
    # Save some time by not doing calc every timestep
    dt_tau = dt*(1/tau)
    sqrtdt_tau = np.sqrt(dt)*(1/tau)

    # Loop through time
    for m in range(M):
        # white noise process
        x = np.random.randn(sysDim+1,1)
        # Recurrent + bias input
        In = mu + W@r_n
        # Euler step
        r_n = r_n + dt_tau*(-r_n + In) + sqrtdt_tau*(DMatrix@x)
        R_n[:,[m+1]] = r_n
    
    # Create the time vector (units: seconds)
    ts = np.arange(M+1)   
    timeVec = dt*ts/1000
         
    return R_n, timeVec      
    
def xcov(x,y):
    # Computes the (bias-adjusted) cross-covariance of two signals
    # Function inputs
    #   x: signal 1
    #   y: signal 2
    xbar = np.mean(x)
    ybar = np.mean(y)
    n = len(x)
    d = np.arange(n)
    k = np.append(d[::-1], d[1:])
    
    cov = signal.correlate(x-xbar, y-ybar, mode='full', method='fft')/(n-k)
    lags = signal.correlation_lags(len(y-ybar), len(x-xbar))
    return cov, lags 