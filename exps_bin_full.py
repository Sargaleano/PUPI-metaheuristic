## Import libraries ##
from pupi_class import PupiBinary
from pupi_bm import *
import numpy as np
import time
import csv

## General settings ##
np.random.seed(int(time.time()))  # (91680801)  # Set random state for reproducibility
problems = [oneMax, squareWave, binVal]
dims = [16, 25, 36, 64, 100]

popsize = [20, 40, 80]
nrates = [.25, .50, .75]
evals = [10000, 20000, 40000]
alphas = [.01, .1, .5]
sigmas = [.01, .1, .5]
stats = []; nreps = 10

for problem in problems:
    for dim in dims:
        for nrep in range(nreps):
            pupi = PupiBinary(fcost=problem, d=dim, viz=False)
            pupi.optimise()
            stats.append(pupi.getResults())
            pupi.summary()
print(stats)

with open('results/exps_dims.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(stats)
writeFile.close()
#
# for problem in problems:
#     for n in popsize:
#         for nrep in range(nreps):
#             pupi = PupiBinary(fcost=problem, d=64, n=n, viz=False)
#             results = pupi.optimise()
#             stats.append(results)
#             pupi.summary()
# print(stats)
# #
# with open('csv/exps_n.csv', 'a') as writeFile:
#   writer = csv.writer(writeFile)
#   writer.writerows(stats)
# writeFile.close()
# ## save_csv(stats, "exps_dim.csv")
# #cwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
#
#
#
# ## save_csv(stats, "exps_dim.csv")
#
# for problem in problems:
#     for n in popsize:
#         for nrep in range(nreps):
#             pupi = PupiBinary(fcost=problem, d=100, n=n, viz=False)
#             results = pupi.optimise()
#             stats.append(results)
#             pupi.summary()
# print(stats)
#
# with open('csv/exps_n100.csv', 'a') as writeFile:
#     writer = csv.writer(writeFile)
#     writer.writerows(stats)
# writeFile.close()
# ## save_csv(stats, "exps_dim.csv")