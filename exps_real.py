## Import libraries ##
import pupi_base as pupi1
from pupi_class import PupiReal
from pupi_bm import *
import numpy as np
import time
import matplotlib.pyplot as plt

## General settings ##
np.random.seed(int(str(int(time.time() * 1000))[-8:-1]))  # (91680801)  # Set random state for reproducibility
problems = [rastrigin, sphere, rosenbrock, himmelblau]
LB = np.array([-5., -5.]); UB = np.array([5., 5.])
stats1 = []; stats2 = []; nreps = 5

## Pupi-Base experiments ##
for _ in range(nreps):
    for problem in problems:
        results = pupi1.optimise(n=20, nw=.25, max_iter=2000, alpha=0.01, sigma=0.1, fcost=problem, LB=LB, UB=UB)
        stats1.append(results)
    results = pupi1.optimise(n=40, nw=.25, max_iter=1000, alpha=0.005, sigma=50, fcost=eggholder, LB=np.array([-512., -512.]), UB=np.array([512., 512.]))
    stats1.append(results)

## Pupi-Class experiments ##
for k in range(nreps):
    for problem in problems:
        pupi2 = PupiReal(fcost=problem)
        pupi2.optimise()
        stats2.append(pupi2.getResults())
        pupi2.summary()
    pupi2 = PupiReal(fcost=eggholder, LB=np.array([-512., -512.]), UB=np.array([512., 512.]), max_eval=40000, n=40, nw=.25, alpha=0.005, sigma=50)
    pupi2.optimise();
    stats2.append(pupi2.getResults());
    pupi2.summary()
    # [plt.plot(results) for results in pupi2.getStats()]; plt.title('PupiReal best/avg./worst: Eggholder (rep=%d)' % k); plt.show()

## Plots Pupi-base ##
plt.hist([results[1] for results in stats1 if results[0]=="eggholder"])
plt.title('Best function cost: Eggholder (Pupi-base)')
plt.show()
plt.plot([results[3] for results in stats1 if results[0]=="eggholder"], marker='o')
plt.title('Execution time secs: Eggholder (Pupi-base)')
plt.show()

## Plots Pupi-class ##
plt.hist([results[2] for results in stats2 if results[0]=="eggholder"])
plt.title('Best function cost: Eggholder (PupiReal)')
plt.show()
plt.plot([results[5] for results in stats2 if results[0]=="eggholder"], marker='o')
plt.title('Execution time secs: Eggholder (PupiReal)')
plt.show()

print("\n------- Stats: PUPI-base --------")
#print(stats1)
print("Best costs (%s): " % "eggholder", [results[1] for results in stats1 if results[0] == "eggholder"])
print("Best times (%s): " % "eggholder", [results[3] for results in stats1 if results[0] == "eggholder"])

print("\n------- Stats: PUPI-class --------")
#print(stats2)
print("Best costs (%s): " % "eggholder", [results[2] for results in stats2 if results[0] == "eggholder"])
print("Best evals (%s): " % "eggholder", [results[3] for results in stats2 if results[0] == "eggholder"])
print("Best times (%s): " % "eggholder", [results[5] for results in stats2 if results[0] == "eggholder"])


# p = PupiReal(fcost=eggholder, LB=np.array([-512.,-512.]), UB=np.array([512.,512.]), max_eval=40000, n=40, alpha=0.005, sigma=20, viz=False)
