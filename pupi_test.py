## Import libraries ##
from pupi_class import PupiReal, PupiBinary
from pupi_bm import *
import numpy as np
import pandas as pd
import time
import csv
import matplotlib.pyplot as plt

np.random.seed(int(str(int(time.time() * 1000))[-8:-1]))  # Set random generator seed
# viz = True        # Visualisation on
viz = False         # Visualisation off
nreps = 5           # Set number of experiment repetitions


#### Continuous-valued experiments ####

# 2D problems #
problems = [rastrigin, rastrigin_offset, rastrigin_bipolar, rosenbrock, himmelblau, sphere]
for i in range(0, nreps):
    for problem in problems:
        pupic = PupiReal(fcost=problem, viz=viz)
        pupic.optimise()
        pupic.summary()

# Eggholder 2D problem has different bounds and therefore step sizes #
    pupic = PupiReal(fcost=eggholder, LB=np.array([-512., -512.]), UB=np.array([512., 512.]), alpha=.9, sigma=50, viz=viz, mode='clipped', nw=.4)
    pupic.optimise()
    pupic.summary()
# quit(1)

# 10D problems #
d = 10
UB = np.repeat(5., d); LB = -UB
problems = [sphere, rastrigin, rastrigin_offset, rastrigin_bipolar]
for problem in problems:
    pupic = PupiReal(fcost=problem, LB=LB, UB=UB, max_eval=100000, n=20, nw=.1, alpha=.1, sigma=1)
    pupic.optimise()
    pupic.summary()

# 30D problems #
d = 30
UB = np.repeat(5., d); LB = -UB
problems = [sphere, rastrigin, rastrigin_offset, rastrigin_bipolar]
for problem in problems:
    pupic = PupiReal(fcost=problem, LB=LB, UB=UB, max_eval=300000, n=20, nw=.1, alpha=.1, sigma=1)
    pupic.optimise()
    pupic.summary()


#### Binary-valued experiments ####

problems = [oneMax, squareWave, powSum] #, binVal]

for i in range(0, nreps):
    # 100D problems #
    for problem in problems:
        pupib = PupiBinary(fcost=problem, d=100, n=40, nw=.1, alpha=.5, sigma=1, max_eval=50000, viz=viz)
        pupib.optimise()
        pupib.summary()

    # 400D problems #
    for problem in problems:
        pupib = PupiBinary(fcost=problem, d=400, n=20, nw=.1, alpha=.5, sigma=1, max_eval=100000)
        pupib.optimise()
        pupib.summary()

#### Binary-Knapsack experiments ####
problems = [knapsack_neglect,knapsack_penalty]

def knapsack_instance(filename):
    myfile = open(filename)
    mytxt = myfile.readline().split()
    n_items= mytxt[0]
    capacity_knapsack = mytxt[1]
    data=np.loadtxt(filename,skiprows=1,dtype=int)
    profit_item=data[:,0]
    weight_item=data[:,1]
    print("the solution for the problem with this parameters","#_items=",n_items,"//n""capacity_knapsack=",capacity_knapsack,"weight_item=",weight_item,"profit_item=",profit_item,"is:")

for problem in problems:
    knapsack_instance('f4_l-d_kp_4_11.txt')
    pupib = PupiBinary(fcost=problem, d=4,n=4, nw=.1, alpha=.5, sigma=1, max_eval=50000, viz=viz, capacity_knapsack=11, weight_item=np.array([2,4,6,7]), profit_item=np.array([6,10,12,13]))
    pupib.optimise()
    pupib.summary()
