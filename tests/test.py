from athena import KernelActiveSubspaces
import numpy as np

np.random.seed(42)
inputs = np.random.uniform(-1, 1, 60).reshape(15, 4)
outputs = np.random.uniform(0, 5, 15)
ss = KernelActiveSubspaces()
ss.compute(inputs=inputs, outputs=outputs, method='local', nboot=49)
ss.partition(2)
inactive = ss.forward(np.random.uniform(-1, 1, 8).reshape(2, 4))[1]
print([inactive])