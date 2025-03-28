#%%
from algorithmes import *
from tools import *
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
import numpy as np

#%%
# load the model
A = load_sparse_csr('Afwd_318')
normA = svds(A, k=1, return_singular_vectors = False)[0]
A = A / normA
normA = 1
capteurs = np.load('sensor_array.npz')
capteurs = capteurs['arr_0']

# define the phantom
N = 318
L = 30e-3

u = read_transparent_png('vaisseau_photo.png')
u = 255 - np.mean(u, axis=2)
u = u[:-1:2,:-1:2]
u = np.array(u,dtype=np.float32) / 255

plt.imsave('resultats/phantom.png', np.uint8(255*rescale_image(u)), cmap ='gray')

#%%
s = A.dot(u.reshape(-1)) 
sigma = 0.01 * np.max(np.abs(s))
s = s + sigma*np.random.randn(s.shape[0])

# algorithm paramaters
beta = normA**2
gamma = 1.9/beta # gradient step size
l = 0.001 #regularisation parameter
epsilon = 1e-5 # stopping criterion
#%%
    
# constructing the observations
fig = plt.figure(frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(u, extent=[-L/2,L/2, -L/2,L/2], cmap='gray')
ax.scatter(capteurs[0,:,1:], capteurs[1,:,1:], marker='o', s = 40)
fig.savefig('resultats/phantom_geometry.png')

#%%
# Rétroprojection
tic_bp = time.time()
recon_bp = (A.T @ s).reshape((N, N))
toc_bp = time.time()
time_bp = toc_bp - tic_bp
a = np.sum(recon_bp * (u - np.mean(u))) / np.sum( recon_bp * (recon_bp - np.mean(recon_bp)) )
b = np.mean(u) - a*np.mean(recon_bp)

# measuring the reconstruction quality
print('SNR pour Rétroprojection: '+f'{SNR(u,np.minimum(np.maximum(a*recon_bp+b,0),1))} dB')
print('####')

# afficher les résultats
plt.figure()
plt.imshow(recon_bp)
plt.title('Rétroprojection')
plt.show()
plt.imsave('resultats/recon_bp.png', np.uint8(255*rescale_image(np.minimum(np.maximum(a*recon_bp+b,0),1))), cmap ='gray')

#%%
# Using Moindres carrés
tic_lsqr = time.time()
recon_ls = moindre_carres(A, s)
toc_lsqr = time.time()
time_lsqr = toc_lsqr - tic_lsqr
recon_ls = np.reshape(recon_ls[0],(N,N))

# measuring the reconstruction quality
print('SNR pour least square : '+f'{SNR(u,np.minimum(np.maximum(recon_ls,0),1))} dB')
print('####')

# afficher les résultats
plt.figure()
plt.imshow(recon_ls)
plt.title('Moindres carrés')
plt.show()
plt.imsave('resultats/recon_ls.png', np.uint8(255*rescale_image(np.minimum(np.maximum(recon_ls,0),1))), cmap ='gray')

#%%
# Using Moindres carrés non negative 
recon_nls, cost_p, tab_snr_nls, tab_tmps_nls = moindre_carres_nonneg(A, s, gamma, N, N, n_iter = 300, x0 = recon_ls.reshape(-1), ref = u)
recon_nls = np.reshape(recon_nls,(N,N))

# measuring the reconstruction quality
print('SNR pour Moindres carrés non négative : '+f'{SNR(u,np.minimum(np.maximum(recon_nls,0),1))} dB')
print('####')

# afficher les résultats
plt.figure()
plt.imshow(recon_nls)
plt.title('Moindres carrés non négative')
plt.show()

plt.figure()
plt.plot(cost_p)
plt.title('Moindres carrés criterion')
plt.show()
plt.imsave('resultats/recon_nls.png', np.uint8(255*rescale_image(np.minimum(np.maximum(recon_nls,0),1))), cmap ='gray')

#%%
# l2-TV optim. pb
beta = normA**2
l = 1e-3
betat = 1e-1
tau = 1/np.sqrt(beta+4*2*betat**2)
sigma = 1/np.sqrt(beta+4*2*betat**2)
theta = 1

recon_tv,cost_tv,cost_tvd,tab_snr_tv,tab_tmps_tv = primal_dual_TV(A, sigma, betat, tau, N, N, s, l, epsilon, n_iter = 1000, theta = 1, x0 = recon_ls.reshape(-1), ref = u)
recon_tv = np.reshape(recon_tv,(N,N))

# measuring the reconstruction quality
print('SNR pour l2-tv : '+f'{SNR(u,recon_tv)} dB')
print('####')

# afficher les résultats
plt.figure()
plt.imshow(recon_tv)
plt.title('l2-tv')
plt.show()

plt.figure()
plt.plot(cost_tv)
plt.title('l2-tv criterion')
plt.show()
plt.imsave('resultats/recon_tv.png', np.uint8(255*rescale_image(recon_tv)), cmap ='gray')

#%%
# l2-Cauchy grad optim. pb (LBFGS)
reg_grad_cauchy = 1e-4 
beta_grad_cauchy = 5e-2
recon_cglb,cost_cglb,tab_snr_cglb,tab_tmps_cglb = LMBFGS_grad(A, s, 200, reg_grad_cauchy, beta_grad_cauchy, N, N, epsilon, normA, u = recon_ls.reshape(-1), ref = u)
recon_cglb = np.reshape(recon_cglb,(N,N))

# measuring the reconstruction quality
print('SNR pour l2-Cauchy grad BFGS : '+f'{SNR(u,np.minimum(np.maximum(recon_cglb, 0),1))} dB')
print('####')

# # afficher les résultats
plt.figure()
plt.imshow(recon_cglb)
plt.title('l2-Cauchy grad BFGS')
plt.show()

plt.figure()
plt.plot(cost_cglb)
plt.title('l2-Cauchy grad BFGS criterion')
plt.show()
plt.imsave('resultats/recon_cauchy_grad_bfgs.png', np.uint8(255*rescale_image(np.minimum(np.maximum(recon_cglb, 0),1))), cmap = 'gray')

#%%
# Plot SNR vs. Time
plt.plot(time_lsqr, SNR(u,np.minimum(np.maximum(recon_ls,0),1)), 'x', markersize=8, label="Moindre carré")
plt.plot(time_bp, SNR(u,np.minimum(np.maximum(a*recon_bp+b,0),1)), 'x', markersize=8, label="Rétroprojection")
plt.plot(tab_tmps_nls, tab_snr_nls, 'b-', label="NonNeg MC")
plt.plot(tab_tmps_tv, tab_snr_tv, 'g-', label="Chambolle-Pock TV")
plt.plot(tab_tmps_cglb, tab_snr_cglb, 'c-', label="LMBFGS-Cauchy")

idx = np.where(tab_snr_cglb>tab_snr_tv[-1])[0]
if len(idx) == 0:
    i_int = len(tab_snr_cglb) - 1
else:
    i_int = idx[0]

plt.plot([tab_tmps_cglb[i_int], tab_tmps_tv[-1]], [tab_snr_cglb[i_int], tab_snr_tv[-1]],
         color='r', linestyle='--')
plt.plot([tab_tmps_cglb[i_int], tab_tmps_cglb[i_int]], [0, tab_snr_cglb[i_int]],
         color='r', linestyle='--')
plt.text(tab_tmps_cglb[i_int], -1.7, f"{tab_tmps_cglb[i_int]:.1f}", ha='center', va='top', fontsize=10)
plt.text(tab_tmps_tv[-1], -1.7, f"{tab_tmps_tv[-1]:.1f}", ha='center', va='top', fontsize=10)

plt.plot(tab_tmps_cglb[i_int], tab_snr_cglb[i_int], 'o')  # BFGS intersection point
plt.plot(tab_tmps_tv[-1], tab_snr_tv[-1], 'o')  # final TV point

plt.xlabel("Temps (s)")
plt.ylabel("SNR (dB)")
plt.title("SNR vs. Temps")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
