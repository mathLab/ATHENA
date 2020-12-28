
import matplotlib.pyplot as plt
import numpy as np
import torch

from athena.active import ActiveSubspaces
from athena.nll import NonlinearLevelSet

torch.set_default_tensor_type(torch.DoubleTensor)


np.random.seed(42)

# global parameters
n_train = 300
n_params = 2

x_np = np.random.uniform(size=(n_train, n_params))
f = x_np[:, 0]**3 + x_np[:, 1]**3 + 0.2 * x_np[:, 0] + 0.6 * x_np[:, 1]
df_np = np.empty((n_train, n_params))
df_np[:, 0] = 3.0*x_np[:, 0]**2 + 0.2
df_np[:, 1] = 3.0*x_np[:, 1]**2 + 0.6 


ss = ActiveSubspaces(1)
ss.fit(inputs=x_np, gradients=df_np)
ss.plot_eigenvalues(figsize=(6, 4))
ss.plot_sufficient_summary(x_np, f, figsize=(6, 4))


nll = NonlinearLevelSet(n_layers=10,
                        active_dim=1, 
                        lr=0.008,
                        epochs=1000,
                        dh=0.25)
x_torch = torch.as_tensor(x_np, dtype=torch.double)
df_torch = torch.as_tensor(df_np, dtype=torch.double)
nll.train(inputs=x_torch,
          gradients=df_torch,
          outputs=f.reshape(-1, 1),
          interactive=True)

# in case of interactive=False
# nll.plot_loss(figsize=(6, 4))
# nll.plot_sufficient_summary(x_torch, f, figsize=(6, 4))

def gridplot(grid_np, Nx=64, Ny=64, color='black', **kwargs):
    grid_1 = grid_np[:, 0].reshape(1, 1, Nx, Ny)
    grid_2 = grid_np[:, 1].reshape(1, 1, Nx, Ny)
    u = np.concatenate((grid_1, grid_2), axis=1)
    
    # downsample displacements
    h = np.copy(u[0, :, ::u.shape[2]//Nx, ::u.shape[3]//Ny])
    # now reset to actual Nx Ny that we achieved
    Nx = h.shape[1]
    Ny = h.shape[2]
    # adjust displacements for downsampling
    h[0, ...] /= float(u.shape[2])/Nx
    h[1, ...] /= float(u.shape[3])/Ny
    # put back into original index space
    h[0, ...] *= float(u.shape[2])/Nx
    h[1, ...] *= float(u.shape[3])/Ny
    
    plt.figure(figsize=(6, 4))
    # create a meshgrid of locations
    for i in range(Nx):
        plt.plot(h[0, i, :], h[1, i, :], color=color, **kwargs)
    for i in range(Ny):
        plt.plot(h[0, :, i], h[1, :, i], color=color, **kwargs)
    for ix, xn in zip([0, -1], ['B', 'T']):
        for iy, yn in zip([0, -1], ['L', 'R']):
            plt.plot(h[0, ix, iy], h[1, ix, iy], 'o', label='({xn},{yn})'.format(xn=xn, yn=yn))
    
    plt.axis('equal')
    plt.legend()
    plt.grid(linestyle='dotted')
    plt.show()


xx = np.linspace(0.0, 1.0, num=8)
yy = np.linspace(0.0, 1.0, num=8)
xxx, yyy = np.meshgrid(xx, yy)
mesh = np.concatenate((np.reshape(xxx, (8**2, 1)), np.reshape(yyy, (8**2, 1))), axis=1)

gridplot(mesh, Nx=8, Ny=8)

grid_torch = nll.forward(torch.from_numpy(mesh))
grid_np = grid_torch.detach().numpy()
gridplot(grid_np, Nx=8, Ny=8)

