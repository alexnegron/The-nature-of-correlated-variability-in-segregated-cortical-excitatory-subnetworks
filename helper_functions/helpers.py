import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.linalg import solve_continuous_lyapunov
from scipy.linalg import expm
from scipy import signal


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

def global_inh_model(t0, r0, r_bar, T, dt, t_kick, kick_bool, kick, tau_E, tau_I, sigE, sigI, c, shared_structure, Wee, Wii, Wei, Wie, alpha):
    # t0: sim start 
    # r0: rates to start at
    # r_bar: desired steady state solution
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
    # alpha: scales E1-E2 connection
    
    M = int(T/dt)
    
    # matrices to store firing rates as row and time as column
    # R_ss = np.zeros((3,1)) # steady-state deterministic solution
    # R_n = np.zeros((3,1)) # stochastic solution
    # R_p = np.zeros((3,1)) # deterministic solution with perturbation
    
    R_ss = np.zeros((3,M+1)) # steady-state deterministic solution
    R_n = np.zeros((3,M+1)) # stochastic solution
    R_p = np.zeros((3,M+1)) # deterministic solution with perturbation
    
    W = np.block([        # weight matrix for global inhibition motif
        [Wee, alpha*Wee, -Wei],
        [alpha*Wee, Wee, -Wei],
        [Wie, Wie, -Wii]
    ])
    
    mu = (np.eye(3)-W)@r_bar # stimulus inputs required for given r_bar
    
    r_n = r0     
    r_ss = r0 
    r_p = r0

    # initial inputs 
    Id = mu 
    In = mu 
    Ip = mu
    
    # white noise process: first three are private noise terms, fourth is shared noise term
    x = np.zeros((4,1)) 
    
    # vector of E/I timescales 
    tau = np.array([[tau_E], [tau_E], [tau_I]])
    
    # noise structure matrix: implements private/shared noise in matrix form
    D = np.diagflat([np.sqrt((1-c)*sigE), np.sqrt((1-c)*sigE), np.sqrt((1-c)*sigI)])
    D_shared = np.array([[np.sqrt(c*sigE)],
                        [np.sqrt(c*sigE)],
                        [np.sqrt(c*sigI)]])
    D_shared = shared_structure*D_shared
    D = np.c_[D,D_shared]

    ts = np.arange(M+1)
    for m in range(M):
    # deterministic sim
        r_ss = r_ss + (dt*(1/tau))*(-r_ss + Id)
        Id = mu + W@r_ss
        # R_ss = np.c_[R_ss, r_ss] # concatenate column
        R_ss[:,[m+1]] = r_ss
    # noise sim
        x = np.random.randn(4,1)
        r_n = r_n + -dt*(1/tau)*r_n + dt*(1/tau)*In + (1/tau)*np.sqrt(2*dt)*(D@x)
        In = mu + W@r_n
        # R_n = np.c_[R_n, r_n]
        R_n[:,[m+1]] = r_n
    # perturbed sim
        if kick_bool:
            if m==t_kick: 
                r_p = r_p + (dt*(1/tau))*(-r_p + Ip) + np.array([[kick],[-kick],[0]])
                Ip = mu + W@r_p
                # R_p = np.c_[R_p, r_p]
            else:
                r_p = r_p + (dt*(1/tau))*(-r_p + Ip)
                Ip = mu + W@r_p
                # R_p = np.c_[R_p, r_p]
            R_p[:,[m+1]] = r_p
                
    return r_ss, r_n, r_p, R_ss, R_n, R_p, W, mu, ts, sigE, sigI


def corr_plot(motif, lowerW_EI, upperW_EI, lowerW_IE, upperW_IE, W_EE, W_II, sigE, sigI, alpha, beta, gamma, zeta, C, b, filename):
    #range of W_EI
    W_EIs = np.linspace(lowerW_EI, upperW_EI, 200)
    W_IEs = np.linspace(lowerW_IE, upperW_IE, 200)
    
    if motif == 3:
        Sigma_private = np.array([
            [np.sqrt((1-C)*sigE), 0, 0, 0],
            [0, np.sqrt((1-C)*sigE), 0, 0],
            [0, 0, np.sqrt((1-C)*sigI), 0]
        ])
        Sigma_shared = np.array([
            [0, 0, 0, np.sqrt(C*sigE)],
            [0, 0, 0, np.sqrt(C*sigE)],
            [0, 0, 0, np.sqrt(C*sigI)*b] # b=1 if all units get shared noise, 0 otherwise
        ])
    elif motif == 4:
        Sigma_private = np.array([
            [np.sqrt((1-C)*sigE), 0, 0, 0, 0],
            [0, np.sqrt((1-C)*sigE), 0, 0, 0],
            [0, 0, np.sqrt((1-C)*sigI), 0, 0],
            [0, 0, 0, np.sqrt((1-C)*sigI), 0]
        ])
        Sigma_shared = np.array([
            [0, 0, 0, 0, np.sqrt(C*sigE)],
            [0, 0, 0, 0, np.sqrt(C*sigE)],
            [0, 0, 0, 0, np.sqrt(C*sigI)*b], 
            [0, 0, 0, 0, np.sqrt(C*sigI)*b]
        ])
        
    else:
        print('input "motif" parameter is either 3 (global inhibition motif) or 4 (specific inhibition motif)')
        
        
    Sigma = Sigma_private + Sigma_shared
    
    K = len(W_EIs)
    N = motif

    # store covs/vars/corrs
    Covs_12 = np.zeros((K,K))
    Corrs_12 = np.zeros((K,K))
    Vars_1 = np.zeros((K,K))
    Vars_2 = np.zeros((K,K))
    Vars_3 = np.zeros((K,K))
    
    Max_evrp = np.zeros((K,K)) # max real part eigenvalue array (for plotting unstable region)

    for i in range(K):
        W_EI = W_EIs[i]
        for j in range(K):
            W_IE = W_IEs[j]
            
            if N == 3: 
                W = np.block([
                    [W_EE, alpha*W_EE, -W_EI],
                    [alpha*W_EE, W_EE, -W_EI],
                    [W_IE, W_IE, -W_II]
                ])
            else: # N=4
                W = np.block([
                    [W_EE, alpha*W_EE, -W_EI, -beta*W_EI],
                    [alpha*W_EE, W_EE, -beta*W_EI, -W_EI],
                    [W_IE, gamma*W_IE, -W_II, -zeta*W_II],
                    [gamma*W_IE, W_IE, -zeta*W_II, -W_II]
                ])
                
            eigvals = np.linalg.eigvals(W)
            evrp = eigvals.real
            max_evrp = np.max(evrp)
            Max_evrp[i,j] = max_evrp

            # compute covariance matrix
            M = -np.eye(N) + W
            Minv = np.linalg.inv(M)
            C = 2*Minv @ Sigma @ Sigma.T @ (Minv.T)
            Corr = correlation_from_covariance(C)

            Cov_12 = Sigma[0,1]
            Corr_12 = Corr[0,1]
            Var1 = Sigma[0,0]
            Var2 = Sigma[1,1]
            Var3 = Sigma[2,2]
            
            Covs_12[i,j] = Cov_12
            
            if max_evrp >= 1:
                Corrs_12[i,j] = np.nan
            else:
                Corrs_12[i,j] = Corr_12
                
            Vars_1[i,j] = Var1 
            Vars_2[i,j] = Var2
            Vars_3[i,j] = Var3
    
    # plotting
    fig,ax = plt.subplots()
    plt.rcParams.update({'hatch.color': 'red'})
    
    divnorm=colors.TwoSlopeNorm(vcenter=0)
    cmap=cm.get_cmap('PRGn')
    cmap = cm.get_cmap("PRGn").copy()
    cmap.set_bad(color='gray')
    
    im=cm.ScalarMappable(norm=divnorm, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='8%', pad=.4)
    im = ax.imshow(np.flip(Corrs_12, axis=0), cmap=cmap, extent=[lowerW_EI, upperW_EI, lowerW_IE, upperW_IE], norm=divnorm)
    cbar=fig.colorbar(im, cax=cax, orientation='vertical')
    
    # use mask to hatch region where network is unstable
    unstable_eigmask = np.ma.masked_less(Max_evrp,1) 
    hatches = ax.contourf(W_EIs, W_IEs, unstable_eigmask, extent=[lowerW_EI, upperW_EI, lowerW_IE, upperW_IE], hatches='//', alpha=0.)
    
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    ax.set_xlabel(r'$W_{IE}$', fontsize=25, labelpad=20)
    ax.set_ylabel(r'$W_{EI}$', rotation=0, fontsize=25, labelpad=35)
    cbar.ax.yaxis.set_tick_params(labelsize=20)
    plt.title(r'Corr$(E_1,E_2)$', fontsize=15)
    
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
    
    
def path_contributions(Wee, Wii, Wie, Wei, sigE1, sigE2, sigI, alpha, nn, ylim_lower, ylim_upper, filename):
    W = np.block([
                    [Wee, alpha*Wee, -Wei],
                    [alpha*Wee, Wee, -Wei],
                    [Wie, Wie, -Wii]
                ])
    
    Eye = np.eye(3)
    c_n = np.diag([np.sqrt(sigE1), np.sqrt(sigE2), np.sqrt(sigI)])
    M = -Eye + W
    D = c_n
    D = np.sqrt(2)*D 
    Minv = np.linalg.inv(M)
    C = Minv @ D @ (np.transpose(Minv@D))
    Corr = correlation_from_covariance(C)
    Corr_E1E2 = Corr[0,1]
    Sigma_E1E2 = C[0,1]
    Delta_inv = np.diag(1 / np.sqrt(np.diag(C)))
    
    WW_cov = np.array([
        [((np.linalg.matrix_power(W,j))@(np.linalg.matrix_power(c_n,2))@np.transpose(np.linalg.matrix_power(W,k)))[0,1] for j in range(nn+1)] for k in range(nn+1)
    ])
    WW_corr = np.array([
        [(Delta_inv@(np.linalg.matrix_power(W,j))@(np.linalg.matrix_power(c_n,2))@np.transpose(np.linalg.matrix_power(W,k))@Delta_inv)[0,1] for j in range(nn+1)] for k in range(nn+1)
    ])
    
    
    CovPaths_by_order = [np.sum(np.diagonal(WW_cov[::-1], offs)) for offs in range(-WW_cov.shape[0]+1, WW_cov.shape[1])]
    
    
    CorrPaths_by_order = [2*np.sum(np.diagonal(WW_corr[::-1], offs)) for offs in range(-WW_corr.shape[0]+1, WW_corr.shape[1])]
    
    
    approx_Sigma = 2*np.sum([(np.linalg.matrix_power(W,j))@(np.linalg.matrix_power(c_n,2))@(np.transpose(np.linalg.matrix_power(W,k))) for j in range(nn+1) for k in range(nn+1)],0)
    approx_Corr = 2*np.sum([Delta_inv@(np.linalg.matrix_power(W,j))@(np.linalg.matrix_power(c_n,2))@(np.transpose(np.linalg.matrix_power(W,k)))@Delta_inv for j in range(nn+1) for k in range(nn+1)],0)
    
    fig, ax = plt.subplots(figsize=(12,4))
    ax.spines.left.set_linewidth(2)
    ax.spines.bottom.set_linewidth(2)
    ax.spines.right.set_color('none')
    ax.spines.top.set_color('none')
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    
    We1e1 = We2e2 = Wee
    We1e2 = We2e1 = alpha*Wee
    We1i = We2i = Wei
    Wie1 = Wie2 = Wie
    
    # parse paths in terms of order 
    inh_order_2 = 2*(-We1i*Wie2 + We1i*We2i - We2i*Wie1)
    exc_order_2 = 2*(We1e1*We1e2 + We1e2*We2e2 + We1e1*We2e1 + We1e2*We2e2 + We1e1*We2e1 + We2e1*We2e2)
    
    inh_order_3 = 2*(-We1e2*We1i*Wie1 - We1e1*We1i*Wie2 - We1i*We2e2*Wie2 - 
                   We1e2*We2i*Wie2 + We1i*Wie2*Wii+We1e1*We1i*We2i + We1e2*We2i**2 - We1i*We2e1*Wie1 - We1i*We2e2*Wie2 - We1i*We2i*Wii + (We1i**2)*We2e1 + We1i*We2e2*We2i - We1e1*We2i*Wie1 - We1e2*We2i*Wie2 - We1i*We2i*Wii -We1i*We2e1*Wie1 - We1e1*We2i*Wie1 - We2e2*We2i*Wie1 - We2e1*We2i*Wie2 + We2i*Wie1*Wii)
    exc_order_3 = 2*((We1e1**2)*We1e2 +(We1e2**2)*We2e1 + We1e1*We1e2*We2e2 + We1e2*(We2e2**2)
    + (We1e1**2)*We2e1 + We1e2*(We2e1**2) + We1e1*We1e2*We2e2 + We1e2*(We2e2**2)
    + (We1e1**2)*We2e1 + (We1e2**2)*We2e1 + We1e1*We2e1*We2e2 + We1e2*(We2e2**2)
    + (We1e1**2)*We2e1 + We1e2*(We2e1**2) + We1e1*We2e1*We2e2 + We2e1*(We2e2**2))
    
    ind = [x for x, _ in enumerate(range(1,len(CorrPaths_by_order)+1))]
    barsum = 0
    for i in range(len(CorrPaths_by_order)):
        idx = ind[i]
        if i == 0:
            pass
        
        elif i==1: # first order path
            corr = CorrPaths_by_order[i]
            barsum += corr
            if corr < 0:
                col = 'mediumorchid'
            else:
                col = 'mediumseagreen'
            order1 = ax.bar(idx, corr, color=col, width=0.35, edgecolor='orangered', hatch='x', linewidth=3.5)
            ax.bar_label(order1, labels=[r'$E$'], label_type='edge', fontsize=22)
        
        elif i == 2: # second order paths 
            corr_through_I = inh_order_2/(np.sqrt(C[0,0])*np.sqrt(C[1,1])) # capture correlation contribution via paths through I
            corr_Not_through_I = exc_order_2/(np.sqrt(C[0,0])*np.sqrt(C[1,1])) # capture correlation contribtion via paths that do /not/ involve I 
            
            # coloring based on whether contribution involves I or not
            if corr_through_I > 0:
                thru_I_col = 'mediumseagreen'
            else:
                thru_I_col = 'mediumorchid'
            
            if corr_Not_through_I > 0:
                not_thru_I_col = 'mediumseagreen'
            else:
                not_thru_I_col = 'mediumorchid'
            
            # form bars
            thru_I = ax.bar(idx-(.35/2), corr_through_I, width=0.35, color=thru_I_col, edgecolor='blue', hatch='x', linewidth=3.5)
            not_thru_I = ax.bar(idx+(.35/2), corr_Not_through_I, width=0.35, color=not_thru_I_col, edgecolor='orangered', hatch='x', linewidth=3.5)
            
            # label bars
            ax.bar_label(thru_I, labels=[r'$I$'], label_type='edge', fontsize=22, padding=7)
            ax.bar_label(not_thru_I, labels=[r'$E$'], label_type='edge', fontsize=22)
            barsum += corr_through_I + corr_Not_through_I
        
        elif i == 3: # third order paths 
            corr_through_I = inh_order_3/(np.sqrt(C[0,0])*np.sqrt(C[1,1]))
            corr_Not_through_I = exc_order_3/(np.sqrt(C[0,0])*np.sqrt(C[1,1]))
            if corr_through_I > 0:
                thru_I_col = 'mediumseagreen'
            else:
                thru_I_col = 'mediumorchid'
            
            if corr_Not_through_I > 0:
                not_thru_I_col = 'mediumseagreen'
            else:
                not_thru_I_col = 'mediumorchid'
            
            thru_I = ax.bar(idx-(.35/2), corr_through_I, width=0.35, color=thru_I_col, edgecolor='blue', hatch='x', linewidth=3.5)
            not_thru_I = ax.bar(idx+(.35/2), corr_Not_through_I, width=0.35, color=not_thru_I_col, edgecolor='orangered', hatch='x', linewidth=3.5)
        
            ax.bar_label(thru_I, labels=[r'$I$'], label_type='edge', fontsize=22, padding=7)
            ax.bar_label(not_thru_I, labels=[r'$E$'], label_type='edge', fontsize=22)
            barsum += corr_through_I + corr_Not_through_I
        
        else: # parsing by path order is done only for order 1, 2, 3 paths—higher order parsings are unwieldy, so just plot the bars
            corr = CorrPaths_by_order[i]
            barsum += corr
            if corr < 0:
                col = 'mediumorchid'
            else:
                col = 'mediumseagreen'
            ax.bar(idx, corr, color=col, edgecolor='black', width=0.7)
    
    ax.axhline(y=0, color="black", linestyle="-")
    
    # following print statements can be uncommented as checks that things work (in particular, the sum of the bars should equal the total correlation
    
    #print(f"\nmax absolute eigenvalue: {np.max(np.absolute(np.linalg.eigvals(W))):4f}\n")
    #print(f"total covariance: {Sigma_E1E2:4f}")
    #print(f"total correlation: {Corr_E1E2:4f}")
    #print(f"covariance approximated to N = {nn} terms: {approx_Sigma[0,1]:4f}")
    #print(f"correlation approximated to N = {nn} terms: {approx_Corr[0,1]:4f}")
    #print(f"bar sum: {barsum}") # this is the sum of the bars
    
    ax.set_xlabel('Path order', fontsize=30)
    ax.set_ylabel(r'$Corr(E_1,E_2)$ contribution', fontsize=30)
    ax.set_xlim(1,3)
    ax.set_ylim(ylim_lower, ylim_upper)
    ax.set_xticks(ind)
    
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    

    
def eos_diagslice(alpha, sigE1, sigE2, sigI, Wii, Wee, Wei_min, Wei_max, filename):
    Weis = np.linspace(Wei_min, Wei_max, 500)
    Wies = np.linspace(Wei_min, Wei_max, 500) # ratio Wei/Wie = 1 
    
    c_n = np.diagflat([np.sqrt(sigE1), np.sqrt(sigE2), np.sqrt(sigI)])
    
    rp_evs = np.zeros((3,0))
    eigvals = np.zeros((3,0))
    corrs = []
    
    for i in range(len(Weis)):
        Wei = Weis[i]
        Wie = Wies[i]

        W = np.block([
                    [Wee, alpha*Wee, -Wei],
                    [alpha*Wee, Wee, -Wei],
                    [Wie, Wie, -Wii]
                ])
        
        # compute eigenvalues of W
        
        eigvals = np.linalg.eigvals(W)
        real_part_evs = (eigvals.real)[::-1]
        real_part_evs = np.expand_dims(real_part_evs, axis=1)
        
        
        # compute covariance matrix
        M = -np.eye(3,3) + W
        Minv = np.linalg.inv(M)
        D = c_n
        D = np.sqrt(2)*D 
        C = Minv @ D @ ((Minv@D).T)
        Delta_inv = np.diag(1 / np.sqrt(np.diag(C)))
        Corr = Delta_inv @ C @ Delta_inv
        
        
        corrs.append(Corr[0,1])
        rp_evs = np.hstack((rp_evs, real_part_evs))
    
    xval_unstable = 0
    j=0
    for i in range(len(Weis)):
        j+=1
        if rp_evs[1,i]-1 < 0:
            xval_unstable = Weis[i]
            break
        else:
            pass
    
    xval_negcorr = 0
    for i in range(j,len(Weis)):
        if corrs[i] < 0:
            xval_negcorr = Weis[i]
            break
        else:
            pass
        

    fig, (ax1, ax2) = plt.subplots(2, figsize=(8,8))
    plt.rcParams.update({'hatch.color': 'red'})


    ax1.plot(Weis[j:], corrs[j:], linewidth=5, color="black")
    ax1.axvspan(Wei_min, xval_unstable, ymin=-1, ymax=1, alpha=0.5, color='gray')
    ax1.axvspan(Wei_min, xval_unstable, ymin=-1, ymax=1, hatch='\\', alpha=0)
    
    ax1.axvspan(xval_unstable, xval_negcorr, ymin=-1, ymax=1, alpha=.2, color='g')
    ax1.axvspan(xval_negcorr, Wei_max, ymin=-1, ymax=1, alpha=.2, color='purple')
    ax1.axhline(y=0, color='black', linestyle='--')
    ax1.set_xlim(Wei_min, Wei_max)
    
    #ax2.plot(wees, rp_evs[0,:])
    #ax3.axhline(y=1, color="r", linestyle="--")
    #ax3.set_ylim(0,2)
    #ax3.plot(Weis, rp_evs[0,:])

    #ax3.plot(wees, rp_evs[1,:])
    ax2.axhline(y=1, color="black", linestyle="--", linewidth=2.5)
    ax2.set_ylim(0.3,1.2)
    ax2.plot(Weis, rp_evs[0,:], linewidth=5, label=r'$\lambda_1$', color='blue')
    ax2.plot(Weis, rp_evs[1,:], linewidth=5, label=r'$\lambda_2$', color='orange')
    ax2.plot(Weis, rp_evs[2,:], linewidth=5, label=r'$\lambda_3$', color='magenta')
    
    ax2.set_xlim(Wei_min, Wei_max)
    
    ax1.spines.left.set_linewidth(2)
    ax1.spines.bottom.set_linewidth(2)
    ax1.spines.right.set_color('none')
    ax1.spines.top.set_color('none')
    ax1.xaxis.set_tick_params(labelsize=30)
    ax1.yaxis.set_tick_params(labelsize=30)
    ax1.set_ylabel(r'Corr$(E_1,E_2)$', fontsize=25, labelpad=20)
    
    ax2.spines.left.set_linewidth(2)
    ax2.spines.bottom.set_linewidth(2)
    ax2.spines.right.set_color('none')
    ax2.spines.top.set_color('none')
    ax2.xaxis.set_tick_params(labelsize=30)
    ax2.yaxis.set_tick_params(labelsize=30)
    ax2.set_xlabel(r'$W_{EI}=W_{IE}$', rotation=0, fontsize=25, labelpad=35)
    ax2.set_ylabel(r'Re$(\lambda)$', fontsize=25, labelpad=15)
    ax2.legend(bbox_to_anchor=(.7,.7), loc="upper left", fontsize=20)

    
    fig.tight_layout()
    plt.savefig(filename)
    plt.show()

def stationary_acovf(lags, M, sigma):
    nonpos_lags = lags[lags<=0]
    pos_lags = lags[lags>0]
    
    arr1 = [sigma @ expm((-M.T)*s) for s in nonpos_lags]
    arr2 = [expm(M*s) @ sigma for s in pos_lags]

    return np.append(arr1,arr2, axis=0)

def auto_corr(Wei, Wie, Wee, Wii, alpha, sigE1, sigE2, sigI, C, tau, shared_structure, r_bar, dt, lags):

    c_n = np.diagflat([np.sqrt((1-C)*sigE1), np.sqrt((1-C)*sigE2), np.sqrt((1-C)*sigI)])
    c_shared = np.array([[np.sqrt(C*sigE1)],
                        [np.sqrt(C*sigE2)],
                        [np.sqrt(C*sigI)]])
    c_shared = shared_structure*c_shared
    c_n = np.c_[c_n, c_shared]
    
    W = np.block([
            [Wee, alpha*Wee, -Wei],
            [alpha*Wee, Wee, -Wei],
            [Wie, Wie, -Wii]
        ])
    mu_0 = (np.eye(3)-W)@r_bar
    
    L = np.eye(3,3)
    LW = L@W
    Eye = np.eye(3,3)
    M = (1/tau)*(-Eye + LW)
    D = (1/tau)*np.sqrt(2)*c_n
    Sigma = solve_continuous_lyapunov(-M, D@(D.T))
    
    acf = stationary_acovf(lags, M, Sigma)

    y11 = [x[0,0] for x in acf]
    y22 = [x[0,0] for x in acf]
    y1 = [x[0,0] for x in acf]
    y2 = [x[0,1] for x in acf]
    y3 = [x[0,2] for x in acf]
    
    return lags, y1,y2,y3, Wei # plot <r_E1(t), r_E2(t+h)>

def compute_corr12(Wee, Wii, Wei, Wie, alpha, zeta, beta, gamma, sigE, sigI, c):
    W = np.array([
                    [Wee, alpha*Wee, -Wei, -beta*Wei],
                    [alpha*Wee, Wee, -beta*Wei, -Wei],
                    [Wie, gamma*Wie, -Wii, -zeta*Wii],
                    [gamma*Wie, Wie, -zeta*Wii, -Wii]
                ])
    
    M = -np.eye(4) + W
    Minv = np.linalg.inv(M)
    c_n = np.diagflat([np.sqrt(sigE), np.sqrt(sigE), np.sqrt(sigI), np.sqrt(sigI)])
    
    D = np.diagflat([np.sqrt((1-c)*sigE), np.sqrt((1-c)*sigE), np.sqrt(sigI), np.sqrt(sigI)])
    D_shared = np.array([[np.sqrt(c*sigE)],
                        [np.sqrt(c*sigE)],
                        [0],
                        [0]]) # no shared noise to I by default
    D = np.c_[D,D_shared]
    D = np.sqrt(2)*D
    
    C = Minv@D@((Minv@D).T)
    Corr = correlation_from_covariance(C)
    return Corr[0,1]

def zeta_beta_gamma(Wee, Wii, Wei, Wie, alpha, c, sigE, sigI, filename1, filename2):
    
    zetas = np.linspace(0,1,100)
    betas = np.linspace(0,1,100)
    gammas = np.linspace(0,1,100)

    corr_z = [] # turn on zeta 
    corr_b = [] # turn on beta
    corr_g = [] # turn on gamma

    corr_zb = [] # turn on zeta and beta, 1:1
    corr_zg = [] # turn on zeta and gamma, 1:1 
    corr_bg = [] # turn on beta and gamma, 1:1 
    
    for i in range(len(zetas)): # turn params on 1 by 1
        
        # turning on zeta alone 
        zeta_z = zetas[i]
        beta_z = 0
        gamma_z = 0
        corr12_z = compute_corr12(Wee, Wii, Wei, Wie, alpha, zeta_z, beta_z, gamma_z, sigE, sigI, c)
        corr_z.append(corr12_z)
        if i==0:
            start = corr12_z
        
        # turning on beta alone 
        zeta_b = 0
        beta_b = betas[i]
        gamma_b = 0
        corr12_b = compute_corr12(Wee, Wii, Wei, Wie, alpha, zeta_b, beta_b, gamma_b, sigE, sigI, c)
        corr_b.append(corr12_b)
        
        # turning on gamma alone 
        zeta_g = 0
        beta_g = 0
        gamma_g = gammas[i]
        corr12_g = compute_corr12(Wee, Wii, Wei, Wie, alpha, zeta_g, beta_g, gamma_g, sigE, sigI, c)
        corr_g.append(corr12_g)

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(22,4),nrows=1, ncols=3, sharey=True)
    for ax in [ax1,ax2,ax3]:
        ax.spines.left.set_linewidth(2)
        ax.spines.bottom.set_linewidth(2)
        ax.spines.right.set_color('none')
        ax.spines.top.set_color('none')
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
    
    ax1.plot(zetas, corr_z, linewidth=3.5, color='blue', label=r'$\zeta$')
    ax1.set_xlabel(r'$\zeta$', fontsize=20, labelpad=5)
    ax1.set_ylabel(r'Corr$(E_1,E_2)$', fontsize=20, labelpad=5)
    ax1.set_xlim(0,1)

    ax2.plot(betas, corr_b, linewidth=3.5, color='purple', label=r'$\beta$')
    ax2.set_xlabel(r'$\beta$', fontsize=20, labelpad=5)
    ax2.set_xlim(0,1)

    ax3.plot(gammas, corr_g, linewidth=3.5, color='orangered', label=r'$\gamma$')
    ax3.set_xlabel(r'$\gamma$', fontsize=20, labelpad=5)
    ax3.set_xlim(0,1)

    ax1.axhline(y=start, linestyle='--', color='mediumseagreen')
    ax2.axhline(y=start, linestyle='--', color='mediumseagreen')
    ax3.axhline(y=start, linestyle='--', color='mediumseagreen')

    plt.savefig(filename1, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    
    ratios = [4,2,1,.5]
    colors = ['seagreen','magenta','black','purple']
            
    corr_zg = [] 
    corr_bg = [] 
    
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(22,4),nrows=1, ncols=3, sharey=True)
    for ax in [ax1,ax2,ax3]:
        ax.spines.left.set_linewidth(2)
        ax.spines.bottom.set_linewidth(2)
        ax.spines.right.set_color('none')
        ax.spines.top.set_color('none')
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        


    
    # zeta/beta plot
    for k in range(len(ratios)): # one line for each ratio
        corr_zb = []
        r = ratios[k]
        col = colors[k]
        
        if r >= 1:
            d = 1
        else:
            d = int(1/r)
            
        for i in range(len(zetas)//d): # zeta : beta             
            if r >= 1: 
                zeta_zb = zetas[i]
                beta_zb = (1/r)*zeta_zb
            else:
                zeta_zb = zetas[i]
                beta_zb = d*zeta_zb
                
            gamma_zb = 0

            corr12_zb = compute_corr12(Wee, Wii, Wei, Wie, alpha, zeta_zb, beta_zb, gamma_zb, sigE, sigI, c)
            corr_zb.append(corr12_zb)
            
            if i==0:
                start = corr12_zb
            ax1.axhline(y=start, linestyle='--', color='mediumseagreen')
        
        ax1.plot(zetas[:len(zetas)//d], corr_zb, linewidth=3.5, color=col)
        
    ax1.set_ylabel(r'Corr$(E_1,E_2)$', fontsize=20, labelpad=5)
    ax1.set_xlabel(r'$\zeta$', fontsize=20, labelpad=5)
    
    
    # zeta/gamma plot 
    for k in range(len(ratios)): 
        corr_zg = []
        r = ratios[k]
        col = colors[k]
        
        if r >= 1:
            d = 1
        else:
            d = int(1/r)
            
        for i in range(len(zetas)//d): # zeta : beta             
            if r >= 1: 
                zeta_zg = zetas[i]
                gamma_zg = (1/r)*zeta_zg
            else:
                zeta_zg = zetas[i]
                gamma_zg = d*zeta_zg
            beta_zg = 0

            corr12_zg = compute_corr12(Wee, Wii, Wei, Wie, alpha, zeta_zg, beta_zg, gamma_zg, sigE, sigI, c)
            corr_zg.append(corr12_zg)
            
            if i==0:
                start = corr12_zg
            ax2.axhline(y=start, linestyle='--', color='mediumseagreen')
        
        ax2.plot(zetas[:len(zetas)//d], corr_zg, linewidth=3.5, color=col)
    ax2.set_xlabel(r'$\zeta$', fontsize=20, labelpad=5)
    
    # beta/gamma plot 
    for k in range(len(ratios)): 
        corr_bg = []
        r = ratios[k]
        col = colors[k]
        
        if r >= 1:
            d = 1
        else:
            d = int(1/r)
            
        for i in range(len(betas)//d): # zeta : beta             
            if r >= 1: 
                beta_bg = betas[i]
                gamma_bg = (1/r)*beta_bg
            else:
                beta_bg = betas[i]
                gamma_bg = d*beta_bg
            zeta_bg = 0

            corr12_bg = compute_corr12(Wee, Wii, Wei, Wie, alpha, zeta_bg, beta_bg, gamma_bg, sigE, sigI, c)
            corr_bg.append(corr12_bg)
            
            if i==0:
                start = corr12_bg
            ax3.axhline(y=start, linestyle='--', color='mediumseagreen')
        
        ax3.plot(betas[:len(betas)//d], corr_bg, linewidth=3.5, color=col)
    ax3.set_xlabel(r'$\beta$', fontsize=20, labelpad=5)
    
    
    plt.tight_layout()
    plt.savefig(filename2, bbox_inches='tight')
    plt.show()