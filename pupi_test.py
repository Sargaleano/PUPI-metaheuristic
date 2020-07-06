## Import libraries ##
from pupi_class import PupiBinary
from pupi_bm import *
import numpy as np
import time
import matplotlib.pyplot as plt

## General settings ##
np.random.seed(int(str(int(time.time() * 1000))[-8:-1]))  # (91680801)  # Set random state for reproducibility
problems = [oneMax, squareWave, binVal]
stats = []; nreps = 10

## PupiBinary experiments ##
for k in range(nreps):
    for problem in problems:
        pupi = PupiBinary(fcost=problem, d=64, nw=.1, viz=False)
        pupi.optimise()
        stats.append(pupi.getResults())
        pupi.summary()
        #[plt.plot(results) for results in pupi.getStats()]; plt.title('PupiReal best/avg./worst: %s (rep=%d)' % (problem.__name__, k)); plt.show()

## Plots PupiBinary ##
if False:
    plt.hist([results[1] for results in stats if results[0]=="oneMax"])
    plt.title('Best function cost: oneMax (PupiBinary)')
    plt.show()
    plt.plot([results[4] for results in stats if results[0]=="oneMax"], marker='o')
    plt.title('Execution time secs: oneMax (PupiBinary)')
    plt.show()

print("\n------- PupiBinary: Summary of results --------")
for problem in problems:
    print("Best costs (%s): " % problem.__name__, [results[2] for results in stats if results[0]==problem.__name__])
    print("Best evals (%s): " % problem.__name__, [results[3] for results in stats if results[0]==problem.__name__])
    print("Best times (%s): " % problem.__name__, [results[5] for results in stats if results[0]==problem.__name__])
#print(stats)

# pupi = PupiBinary(fcost=oneMax, d = 256, n = 256, nw = .25, alpha = 0.1, sigma = 0.1, max_eval = 120000, stats = False)
