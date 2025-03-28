from scipy.sparse.linalg import lsqr
import scipy.optimize
import numpy as np
import time
from tools import SNR

# least squares
def moindre_carres(A, s):
    '''least squares from scipy'''
    recon = lsqr(A, s, atol=0, btol=0, conlim=0,iter_lim = 25)
    return recon
    
# Regularized least squares
# gradient function to be used in the following l2-regularized algorithms
def gradient(u, A, s):
    '''computes the gradient as a function of u(the image)
    of the data fidelity part that is the gradient of 
    (1\2)||s - Au||^2'''
    Aus = A@u.reshape(-1) - s
    return A.T@Aus

# non negative least squares
def moindre_carres_nonneg(A, s, gamma, N1, N2, n_iter, x0 = None, ref=None):
    '''non negative least squares --> projected gradient'''
    if x0 is None:
        recon = np.zeros(np.size(A,1))
    else:
        recon = x0
    F = np.zeros(n_iter)
    tab_snr = np.zeros(n_iter)
    tab_tmps = np.zeros(n_iter)
    tic = time.time()
    
    F[0] = 1/2*np.linalg.norm((s-(A@recon.reshape(-1))))**2
    if ref is not None:
        tab_snr[0] = SNR(ref, recon.reshape((N1, N2)))

    for i in range(n_iter):
        if not(i%10):
            print('Non neg Least square iteration : '+f'{i}/ {n_iter}')
        recon = np.maximum(recon-gamma*gradient(recon,A,s),0)
        F[i] = 1/2*np.linalg.norm((s-(A@recon.reshape(-1))))**2
        toc = time.time()
        tab_tmps[i] = toc - tic
        if ref is not None:
            tab_snr[i] = SNR(ref, recon.reshape((N1, N2)))
    return recon, F, tab_snr, tab_tmps

# TV penalized least squares (Chambolle-Pock primal-dual algorithm)
def grad_col(u):
    '''This function compute the column gradient of a 2D image u'''
    gc = np.zeros(u.shape)
    gc[:, :-1] = u[:, 1:] - u[:, :-1]
    gc[:, -1] = u[:, 0] - u[:, -1]
    return gc

def grad_row(u):
    '''This function compute the row gradient of a 2D image u'''
    gr = np.zeros(u.shape)
    gr[:-1, :] = u[1:, :] - u[:-1, :]
    gr[-1, :] = u[0, :] - u[-1, :]
    return gr

def grad_tot(u):
    '''The total gradient of a 2D image u'''
    N1, N2 = u.shape
    d = u.ndim
    gt = np.zeros((N1, N2, d))
    gt[:, :, 0] = grad_row(u)
    gt[:, :, 1] = grad_col(u)
    return gt

def gradTrans_row(u):
    '''The adjoint of the row gradient'''
    gtr = np.zeros(u.shape)
    gtr[1:, :] = u[:-1, :] - u[1:, :]
    gtr[0, :] = u[-1,:] - u[0,:]
    return gtr

def gradTrans_col(u):
    '''The adjoint of the column gradient'''
    gtc = np.zeros(u.shape)
    gtc[:, 1:] = u[:, :-1] - u[:, 1:]
    gtc[:, 0] = u[:, -1] - u[:, 0]
    return gtc

def gradTrans_tot(grad):
    return gradTrans_row(grad[:, :, 0]) + gradTrans_col(grad[:, :, 1])

def obj_tv(A, u, s, labda, N1, N2):
    '''The Total variation objective'''
    norm = np.linalg.norm
    gradu = grad_tot(u.reshape(N1, N2))
    normu = np.sqrt(gradu[:, :, 0]**2 + gradu[:, :, 1]**2)
    return labda*(normu.sum()) + 0.5*norm(s-(A@u.reshape(-1)))**2

def dual_obj(s, y1):
    '''The dual of the TV objective'''
    norm=np.linalg.norm
    return 0.5*norm(y1)**2 + np.dot(s.ravel(),y1.ravel())

def prox_sigma_f1_star(y1, s, sigma):
    '''The prox of the conjugate of f1 in the augmented 
    TV-regularization reformulation '''
    return (y1 - sigma*s)/(1+sigma)

def prox_sigma_f2_star(labda, beta, y2):
    '''The prox of the conjugate of f2 in the augmented TV-regularization reformulation '''
    normy = np.sqrt(y2[:, :, 0]**2 + y2[:, :, 1]**2)
    dnomi = ((np.maximum((beta/labda)*
                         normy, 1))[:, :, np.newaxis])
    return y2/dnomi

def primal_dual_TV(A, sigma, beta, tau, N1, N2, s, labda, epsilon, n_iter, theta=1, x0=None, ref = None):
    '''Chambolle-Pock implementation using the augmented TV-regularization reformulation'''
    N=N1*N2

    if x0 is None:
        x = np.zeros(N)
    else:
        x = x0
    xbar = x
    y1 = A@x.reshape(-1)
    y2 = beta*grad_tot(x.reshape(N1, N2))
    F = np.zeros(n_iter)
    Fstar = np.zeros(n_iter)
    F[0] = obj_tv(A=A, u=x, s=s, labda=labda,N1=N1, N2=N2)
    Fstar[0] = dual_obj(s, y1)
    x_prev = x

    tab_snr = np.zeros(n_iter)
    tab_tmps = np.zeros(n_iter)
    if ref is not None:
        tab_snr[0] = SNR(ref, np.minimum(np.maximum(x, 0),1).reshape(N1, N2))
    tic = time.time()
    for k in range(n_iter):
        param1 = y1 + sigma*A@xbar.reshape(-1)
        y1 = prox_sigma_f1_star(y1=param1, s=s, sigma=sigma)

        param2 = y2 + sigma*beta*grad_tot(xbar.reshape(N1, N2))
        y2 = prox_sigma_f2_star(labda = labda, beta=beta, y2=param2)

        x_new = (x_prev.reshape(-1) - tau*(A.T@y1 + beta*(gradTrans_tot(y2)).reshape(-1))).reshape(1, -1)
        x_new = np.minimum(np.maximum(x_new, 0),1)
        xbar = x_new + theta*(x_new - x_prev)
        if k<(n_iter-1):
            F[k+1] = obj_tv(A=A, u=x_new, s=s, labda=labda,N1=N1, N2=N2)
            Fstar[k+1] = dual_obj(s, y1)
        x_prev=x_new

        toc = time.time()
        tab_tmps[k] = toc - tic
        if ref is not None:
            tab_snr[k] = SNR(ref, np.minimum(np.maximum(x_new, 0),1).reshape(N1, N2))

        if n_iter<500:
            if not(k%10):
                print('TV iteration : '+f'{k}/ {n_iter}')
        else:
            if not(k%50):
                print('TV iteration : '+f'{k}/ {n_iter}')

    return x_new, F, Fstar, tab_snr, tab_tmps

# 
def grad_cauchy_grad(x,beta):
    Dx = grad_tot(x)
    nDx2 = np.sum(Dx**2,axis=2)[:,:,np.newaxis]
    v = gradTrans_tot( 2 * Dx / (nDx2 + beta**2))
    return v

def LMBFGS_grad(A, s, n_iter, labda, beta, N1, N2, eps, normA, u = None, ref = None):
    N = N1*N2
    k = 0

    if u is None:
        u0 = np.zeros(N)
    else:
        u0 = u
    L = normA**2 + (labda*2)*(4**2/beta)
    
    gradc = lambda x: (A.T@(A@x - s)) + labda*grad_cauchy_grad(x.reshape((N1,N2)), beta).reshape(-1) 
    objc = lambda x: 0.5*np.sum((A@x - s)**2) - labda*np.sum((np.log(beta/(beta**2 + np.sum(grad_tot(x.reshape((N1,N2)))**2,axis=2)))))
        
    gk = gradc(x=u0)
    uk = u0
    F = np.zeros(n_iter)
    
    list_sk = np.zeros((n_iter,N))
    list_yk = np.zeros((n_iter,N))
    
    tab_snr = np.zeros(n_iter)
    tab_tmps = np.zeros(n_iter)
    if ref is not None:
        tab_snr[0] = SNR(ref, np.minimum(np.maximum(uk, 0),1).reshape(N1, N2))
    tic = time.time()

    while np.linalg.norm(gk) > eps and k < n_iter:
        if not(k%10):
            print('l2-Cauchy gradient BFGS iteration : '+f'{k}/ {n_iter}')
        
        Hk= scipy.optimize.LbfgsInvHessProduct(list_sk[:k,:],list_yk[:k,:])
        pk = -Hk.matvec(gk)
        
        fk = objc(x=uk)
        F[k] = fk
        
        uk1 = uk + pk
        sk = uk1 - uk
        gk1 = gradc(x=uk1)
        yk = gk1 - gk
        if np.linalg.norm(gk) < 1:
            alpha = 3
        else:
            alpha = 0.01
        epsil = 1e-6
                
        if (np.dot(yk, sk)/(sk**2).sum()) >= (epsil * np.linalg.norm(gk)**alpha):
            list_sk[k,:] = sk
            list_yk[k,:] = yk
    
        gk = gk1
        uk = uk1

        toc = time.time()
        tab_tmps[k] = toc - tic
        if ref is not None:
            tab_snr[k] = SNR(ref, np.minimum(np.maximum(uk, 0),1).reshape(N1, N2))

        k += 1

    return uk,F, tab_snr, tab_tmps