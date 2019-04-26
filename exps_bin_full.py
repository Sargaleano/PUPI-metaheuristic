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
nwrates = [.25, .50, .75]
evals = [10000, 20000, 40000]
alphas = [.01, .1, .5]
sigmas = [.01, .1, .5]
nreps = 100

## d ##
stats = [];
for problem in problems:
    for dim in dims:
        for nrep in range(nreps):
            pupi = PupiBinary(fcost=problem, d=dim, viz=False)
            pupi.optimise()
            stats.append(pupi.getResults())
            pupi.summary()
#print(stats)

with open('results/bin-dims.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(stats)
writeFile.close()

## popsize ##
stats = [];
for problem in problems:
    for n in popsize:
        for nrep in range(nreps):
            pupi = PupiBinary(fcost=problem, d=64, n=n, viz=False)
            pupi.optimise()
            stats.append(pupi.getResults())
            pupi.summary()
#print(stats)

with open('results/bin-popsize.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(stats)
writeFile.close()

## nwrates ##
stats = [];
for problem in problems:
    for nw in nwrates:
        for nrep in range(nreps):
            pupi = PupiBinary(fcost=problem, d=64, n=40, nw=nw, viz=False)
            pupi.optimise()
            stats.append(pupi.getResults())
            pupi.summary()
#print(stats)

with open('results/bin-nrates.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(stats)
writeFile.close()

## evals; d=36 ##
stats = [];
for problem in problems:
    for max_eval in evals:
        for nrep in range(nreps):
            pupi = PupiBinary(fcost=problem, d=36, n=40, nr=0.25, max_eval=max_eval, viz=False)
            pupi.optimise()
            stats.append(pupi.getResults())
            pupi.summary()
#print(stats)

with open('results/bin-evals-36.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(stats)
writeFile.close()

## evals; d=64 ##
stats = [];
for problem in problems:
    for max_eval in evals:
        for nrep in range(nreps):
            pupi = PupiBinary(fcost=problem, d=64, n=40, nr=0.25, max_eval=max_eval, viz=False)
            pupi.optimise()
            stats.append(pupi.getResults())
            pupi.summary()
#print(stats)

with open('results/bin-evals-64.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(stats)
writeFile.close()

## evals; d=100 ##
stats = [];
for problem in problems:
    for max_eval in evals:
        for nrep in range(nreps):
            pupi = PupiBinary(fcost=problem, d=100, n=80, nr=0.25, max_eval=max_eval, viz=False)
            pupi.optimise()
            stats.append(pupi.getResults())
            pupi.summary()
#print(stats)

with open('results/bin-evals-100.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(stats)
writeFile.close()

## alphas ##
stats = [];
for problem in problems:
    for alpha in alphas:
        for nrep in range(nreps):
            pupi = PupiBinary(fcost=problem, alpha=alpha, viz=False)
            pupi.optimise()
            stats.append(pupi.getResults())
            pupi.summary()
#print(stats)

with open('results/bin-alphas.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(stats)
writeFile.close()

## sigmas ##
stats = [];
for problem in problems:
    for sigma in sigmas:
        for nrep in range(nreps):
            pupi = PupiBinary(fcost=problem, sigma=sigma, viz=False)
            pupi.optimise()
            stats.append(pupi.getResults())
            pupi.summary()
#print(stats)

with open('results/bin-sigmas.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(stats)
writeFile.close()
