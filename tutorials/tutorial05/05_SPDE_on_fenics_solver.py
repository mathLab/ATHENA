from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')


def compute_mesh_map(mesh, dim):
    m_map = np.zeros((dim, 2))
    for j, cell in enumerate(cells(mesh)):
        m_map[j, :] = cell.midpoint().array()[:2]
    # print(m_map.shape)
    return m_map


def compute_cov(mesh, beta, dim, mesh_map):
    print("start covariance assemble")
    cov = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(i, dim):
            cov[j, i] = cov[i, j] = np.exp(
                -(np.linalg.norm(mesh_map[i, :] - mesh_map[j, :], 1)) / (beta))

    print("end covariance assemble")
    evals, evecs = np.linalg.eig(cov)
    E = (evals[:m] * evecs[:, :m]).T
    return cov, E


def set_conductivity(sim_index, mesh, c):
    # print("set conductivity")
    D = FunctionSpace(mesh, "DG", 0)
    kappa = Function(D)
    dm = D.dofmap()
    for i, cell in enumerate(cells(mesh)):
        kappa.vector()[dm.cell_dofs(cell.index())] = np.exp(c[sim_index, i])
    return kappa


def boundary(x):
    return x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS


def boundary0(x):
    return x[0] < DOLFIN_EPS


def compute_solution(sim_index, mesh, kappa, pl=False):
    # print("compute solution")
    V = FunctionSpace(mesh, "Lagrange", 1)
    u0 = Expression("10*x[1]*(1-x[1])", degree=0)
    bc = DirichletBC(V, Constant(0.0), boundary)
    bc0 = DirichletBC(V, u0, boundary0)

    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(
        1.0
    )  #Expression("exp( - 2*pow(x[0]-0.5, 2) - 2*pow(x[1]-0.5, 2) )", element=V.ufl_element())
    a = kappa * inner(grad(u), grad(v)) * dx
    L = f * v * dx
    u = Function(V)
    solve(a == L, u, [bc, bc0])

    if pl:
        u_pl = plot(u, title='u')
        plt.colorbar(u_pl)
        plt.show()
    return u


def restrict(mesh, v):
    # print("restrict on outflow right side")
    Right = AutoSubDomain(lambda x, on_bnd: near(x[0], 1) and on_bnd)
    V = FunctionSpace(mesh, 'CG', 1)
    bc0 = DirichletBC(V, 1, Right)
    u = Function(V)
    bc0.apply(u.vector())
    v_restriction = v.vector()[u.vector() == 1]
    return v_restriction.mean()


def compute_gradients(component_index,
                      mesh,
                      kappa,
                      E,
                      boundary,
                      cache,
                      solution,
                      pl=False):
    # print("compute gradient")
    V = FunctionSpace(mesh, "Lagrange", 1)
    bc = DirichletBC(V, Constant(0.0), boundary)
    w = TrialFunction(V)
    v = TestFunction(V)
    a = kappa * inner(grad(w), grad(v)) * dx
    D = FunctionSpace(mesh, "DG", 0)

    dkappa = Function(D)
    dm = D.dofmap()
    for i, cell in enumerate(cells(mesh)):
        dkappa.vector()[dm.cell_dofs(
            cell.index())] = kappa.vector()[dm.cell_dofs(
                cell.index())] * E[component_index, i]
    rhs = dkappa * inner(grad(solution), grad(v)) * dx

    w = Function(V)
    solve(a == rhs, w, bc)

    if pl:
        w_pl = plot(w, title='w')
        plt.colorbar(w_pl)
        plt.show()
    return w


def show_mode(mode, mesh):
    c = MeshFunction("double", mesh, 2)
    # value = mode.dot(E)

    # Iterate over mesh and set values
    for i, cell in enumerate(cells(mesh)):
        c[cell] = mode[i]  #np.exp(value[i])

    plot(c)
    plt.show()


# Read mesh from file and create function space
mesh = Mesh("data/mesh.xml")

#dim = 6668 #mesh_2
dim = 3194
m = 10
M = 500
d = 1668
cache = np.zeros((d, m))
cache_res = np.zeros(m)

#choose lengthscale
beta = 0.015  #beta=0.03
inputs = np.random.multivariate_normal(np.zeros(m), np.eye(m), M)

#samples
np.save("data/inputs", inputs)

#covariance modes assemble
m_map = compute_mesh_map(mesh, dim)
cov, E = compute_cov(mesh, beta, dim, m_map)
c = inputs.dot(E)
np.save("data/covariance", cov)
np.save("data/cov_modes", E)

print("Karhunen-Loève mode shape", E.shape)

n = 2
print("Mode number {} of Karhunen-Loève decomposition".format(n))
show_mode(E[n, :], mesh)

# cov = np.load("data/covariance.npy", allow_pickle=True)
# E = np.load("data/cov_modes.npy", allow_pickle=True)

V = FunctionSpace(mesh, "Lagrange", 1)
dofs = V.dofmap().dofs()

# Get coordinates as len(dofs) x gdim array
dim = V.dim()
N = mesh.geometry().dim()
dofs_x = V.tabulate_dof_coordinates()
n_dof = 300
print("Coordinates of degree of freedom number {0} are {1}".format(
    n_dof, dofs_x[n_dof]))

mesh = Mesh("data/mesh.xml")
V = FunctionSpace(mesh, "Lagrange", 1)
u = Function(V)
print(np.array(u.vector()[:]).shape)

for j in range(16):
    for i in range(1668):
        if i == (j + 1) * 100:
            u.vector()[i] = 1
        else:
            u.vector()[i] = 0
    plot(u, title='dof {}'.format((1 + j) * 100))
    plt.savefig('data/component_{}.png'.format((1 + j) * 100))

for it in range(M):
    print("Solution number :", it)
    #set conductivity
    kappa = set_conductivity(it, mesh, c)
    #plot(kappa)
    #plt.show()

    #compute solution
    u = compute_solution(it, mesh, kappa, pl=False)  #pl=True to plot
    u_res = restrict(mesh, u)
    #print("mean of the solution restricted on the outflow (right side)", u_res)

    #compute gradients
    for j in range(m):
        #print("Evaluating gradient component number :", j)
        du = compute_gradients(j, mesh, kappa, E, boundary, cache, u)
        du_res = restrict(mesh, du)
        cache[:, j] = du.vector()[:]
        cache_res[j] = du_res

    file = Path("data/outputs.npy")
    with file.open('ab') as f:
        np.save(f, u.vector()[:])

    file = Path("data/outputs_res.npy")
    with file.open('ab') as f:
        np.save(f, u_res)

    file = Path("data/gradients.npy")
    with file.open('ab') as f:
        np.save(f, cache)

    file = Path("data/gradients_res.npy")
    with file.open('ab') as f:
        np.save(f, cache_res)
