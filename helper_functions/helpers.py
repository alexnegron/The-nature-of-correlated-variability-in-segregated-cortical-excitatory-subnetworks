import numpy as np



def threshlin(I):
    return np.maximum(I, 0)

def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def theory_cov(N, k, s, r_ss, W, c_ne, c_ni, tau_n, tau):
    sgn = np.block([
        [np.ones((3*N,N)), np.ones((3*N,N)), -np.ones((3*N,N))]
    ])
    
    c_n = np.diagflat(np.block([
        [np.full((N,1), c_ne)],
        [np.full((N,1), c_ne)],
        [np.full((N,1), c_ni)]
    ]))
    
    L = np.diagflat(d_threshlin(s + (sgn*W)@r_ss, k))
    LW = L@(sgn*W)
    Eye = np.eye(3*N)
    
    M = (1/tau)*(-Eye + LW)
    Minv = np.linalg.inv(M)
    
    D = c_n@L
    D = (1/tau)*np.sqrt(2*tau_n)*D
    
    Sigma = Minv @ D @ ((Minv@D).T)
    
    return Sigma

def global_inh_model(t0, r0, T, dt, t_kick, kick_bool, kick, tau_E, tau_I, sigE, sigI, c, shared_structure, Wee, Wii, Wei, Wie, mu, alpha):
    # t0: sim start 
    # r0: rates to start at
    # T: sim end 
    # dt: size of timestep  
    # t_kick: time step to perturb excitatory rates at 
    # kick_bool: is there a perturbation? T/F
    # kick: magnitude of the perturbation
    # tau_E/tau_I: E/I timescale constant
    # sigE/sigI: E/I noise magnitudes
    # c: scales proportion of shared to private noise, 0 <= c <= 1
    # shared_structure: [1,1,1] = all pops get shared noise, [1,1,0] = only E pops get shared noise
    # Wee/Wii/Wei/Wie: synaptic weights
    # mu: stimulus input 
    # alpha: scales E1-E2 connection
    
    # matrices to store firing rates as row and time as column
    R_ss = np.zeros((3,1)) # steady-state deterministic solution 
    R_n = np.zeros((3,1)) # stochastic solution
    R_p = np.zeros((3,1)) # deterministic solution with perturbation 
    
    W = np.block([
        [Wee, alpha*Wee, -Wei],
        [alpha*Wee, Wee, -Wei],
        [Wie, Wie, -Wii]
    ])
    
    r_n = r0
    r_ss = r0
    r_p = r0

    # initial inputs 
    Id = mu 
    In = mu 
    Ip = mu
    
    # white noise process 
    x = np.zeros((4,1))
    
    # vector of E/I timescales 
    tau = np.array([[tau_E], [tau_E], [tau_I]])
    
    # noise structure matrix
    D = np.diagflat([np.sqrt((1-c)*sigE), np.sqrt((1-c)*sigE), np.sqrt((1-c)*sigI)])
    D_shared = np.array([[np.sqrt(c*sigE)],
                        [np.sqrt(c*sigE)],
                        [np.sqrt(c*sigI)]])
    D_shared = shared_structure*D_shared
    D = np.c_[D,D_shared]

    
    M = int(T/dt)
    ts = np.arange(M+1)
    for m in range(M):
    # deterministic sim
        r_ss = r_ss + (dt*(1/tau))*(-r_ss + Id)
        Id = mu + W@r_ss
        R_ss = np.c_[R_ss, r_ss]
    # noise sim
        x = np.random.randn(4,1)
        r_n = r_n + -dt*(1/tau)*r_n + dt*(1/tau)*In + (1/tau)*np.sqrt(2*dt)*(D@x)
        In = mu + W@r_n
        R_n = np.c_[R_n, r_n]
    # perturbed sim
        if kick_bool:
            if m==t_kick: 
                r_p = r_p + (dt*(1/tau))*(-r_p + Ip) + np.array([[kick],[-kick],[0]])
                Ip = mu + W@r_p
                R_p = np.c_[R_p, r_p]
            else:
                r_p = r_p + (dt*(1/tau))*(-r_p + Ip)
                Ip = mu + W@r_p
                R_p = np.c_[R_p, r_p]
                
    return r_ss, r_n, r_p, R_ss, R_n, R_p, W, mu, ts, sigE, sigI