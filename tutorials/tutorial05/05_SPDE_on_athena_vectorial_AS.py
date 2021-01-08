import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import GPy
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm
from collections import namedtuple
from functools import partial
from pathlib import Path
import os

from athena.active import ActiveSubspaces
from athena.utils import Normalizer

import warnings
warnings.filterwarnings('ignore')

p = Path("data/outputs_res.npy")
with p.open('rb') as fi:
    fsz = os.fstat(fi.fileno()).st_size
    out = np.load(fi)
    while fi.tell() < fsz:
        out = np.vstack((out, np.load(fi)))
f_ = out
print("Output shape", f_.shape)

p = Path("data/outputs.npy")
with p.open('rb') as fi:
    fsz = os.fstat(fi.fileno()).st_size
    out_ = np.load(fi)
    while fi.tell() < fsz:
        out_ = np.vstack((out_, np.load(fi)))
fa_ = out_
print("Output shape", fa_.shape)

p = Path("data/inputs.npy")
with p.open('rb') as fi:
    fsz = os.fstat(fi.fileno()).st_size
    out1 = np.load(fi)
    while fi.tell() < fsz:
        out1 = np.vstack((out1, np.load(fi)))
x_ = out1
print("Input shape", x_.shape)

p = Path("data/gradients_res.npy")
with p.open('rb') as fi:
    fsz = os.fstat(fi.fileno()).st_size
    out2 = np.load(fi)
    while fi.tell() < fsz:
        out2 = np.vstack((out2, np.load(fi)))
df_ = out2
print("Gradients shape", df_.shape)

p = Path("data/gradients.npy")
with p.open('rb') as fi:
    fsz = os.fstat(fi.fileno()).st_size
    out2_ = np.load(fi)
    while fi.tell() < fsz:
        out2_ = np.vstack((out2_, np.load(fi)))
dfa_ = out2_
print("Gradients shape", dfa_.shape)

p = Path("data/metric.npy")
with p.open('rb') as fi:
    fsz = os.fstat(fi.fileno()).st_size
    out3 = np.load(fi)
    while fi.tell() < fsz:
        out3 = np.vstack((out3, np.load(fi)))
metric = out3
print("Metric shape", metric.shape)

#simulation parameters
np.random.seed(42)
n_samples = x_.shape[0]
input_dim = x_.shape[1]
d = fa_.shape[1]
dim = 1

#process data
x, f, df = x_, f_, df_
print("data", x.shape, f.shape, df.shape)

#AS
ss = ActiveSubspaces(dim=1)
ss.fit(inputs=x, outputs=f, gradients=df)
ss.plot_eigenvalues()
ss.plot_eigenvectors()
ss.plot_sufficient_summary(inputs=x, outputs=f)


## Active Subspaces with vectorial outputs
#process data
x, f, df = x_, fa_, dfa_.reshape(n_samples, d, input_dim)
print("data", x.shape, f.shape, df.shape)

#vectorial AS
vss = ActiveSubspaces(dim=5, n_boot=10)
vss.fit( inputs=x, outputs=f, gradients=df, metric=metric)
np.save("data/modes_AS", vss.W1)
vss.dim=1
vss._partition()
vss.plot_eigenvalues()
vss.plot_eigenvectors()

components = [100+j*100 for j in range(16)]
for i in components:
    img = mpimg.imread("data/component_{}.png".format(i))
    plt.imshow(img)
    plt.axis('off')
    vss.plot_sufficient_summary(inputs=x, outputs=f[:, i], title="dof {}".format(i), figsize=(4,4))

