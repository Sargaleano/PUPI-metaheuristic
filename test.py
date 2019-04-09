import pupi_base as pupi
import numpy as np

stats = []
for _ in range(10):
    results = pupi.optimise(n=20, nw=.25, max_iter=2000, alpha=0.01, sigma=0.1, fcost=pupi.rastrigin, LB=np.array([-5., -5.]), UB=[5, 5])
    stats.append(results)
    results = pupi.optimise(n=20, nw=.25, max_iter=2000, alpha=0.01, sigma=0.1, fcost=pupi.sphere, LB=np.array([-5., -5.]), UB=[5, 5])
    stats.append(results)
    results = pupi.optimise(n=20, nw=.25, max_iter=2000, alpha=0.001, sigma=0.1, fcost=pupi.rosenbrock, LB=np.array([-5., -5.]), UB=[5, 5])
    stats.append(results)
    results = pupi.optimise(n=20, nw=.25, max_iter=2000, alpha=0.01, sigma=0.1, fcost=pupi.himmelblau, LB=np.array([-5., -5.]), UB=[5, 5])
    stats.append(results)
    results = pupi.optimise(n=40, nw=.25, max_iter=1000, alpha=0.005, sigma=20, fcost=pupi.eggholder, LB=np.array([-512., -512.]), UB=[512,512])
    stats.append(results)
print(stats)