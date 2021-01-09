6from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
plt.rcParams.update({'font.size': 18})

import warnings
warnings.filterwarnings('ignore')

# Create mesh
mesh = Mesh("data/mesh.xml")
c = MeshFunction("double", mesh, 2)
kar = MeshFunction("double", mesh, 2)

modes= np.load("data/modes_AS.npy")
E = np.load("data/cov_modes.npy")
sample = modes.T.dot(E)

print(modes.shape)
print(E.shape)
print(sample.shape)
sample_k = E[:5, :]

# Iterate over mesh and set values
for j in range(5):
    for i, cell in enumerate(cells(mesh)):
        c[cell] = sample[j, i]
        kar[cell] = sample_k[j, i]

    plt.figure(figsize=(6, 4))
    pl = plot(kar)
    v1 = np.linspace(kar.array().min(), kar.array().max(), 5)
    cb = plt.colorbar(pl,fraction=0.046, pad=0.04, ticks=v1)
    cb.ax.tick_params(labelsize='large')
    cb.ax.set_yticklabels(["{:2.1f}".format(i) for i in v1])
    plt.tight_layout()
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.show()
    # plt.savefig("KAR_"+str(j)+".pdf", bbox_inches='tight')
    # plt.close()

    plt.figure(figsize=(6, 4))
    pl = plot(c)
    v1 = np.linspace(c.array().min(), c.array().max(), 5)
    cb = plt.colorbar(pl,fraction=0.046, pad=0.04, ticks=v1)
    cb.ax.tick_params(labelsize='large')
    cb.ax.set_yticklabels(["{:2.1f}".format(i) for i in v1])
    plt.tight_layout()
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.show()
    # plt.savefig("AS_output_"+output+str(j)+".pdf", bbox_inches='tight')
    # plt.close()

