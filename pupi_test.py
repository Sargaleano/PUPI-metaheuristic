## Import libraries ##
from pupi_class import PupiReal, PupiBinary
from pupi_bm import *
import numpy as np
import time
from timeit import timeit
import pandas as pd
import matplotlib.pyplot as plt
import zipfile

np.random.seed(int(str(int(time.time() * 1000))[-8:-1]))  # Set random generator seed
viz = True        # Visualisation on
# viz = False         # Visualisation off
nreps = 1 #5           # Set number of experiment repetitions

if(False):
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


if(False):
    #### Binary-valued experiments ####

    problems = [oneMax, squareWave, binVal, powSum]

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


if(True):
    #### Combinatorial experiments ####
    problems = [knapsack_discard,#knapsack_penalty
                 ]
    file_tester = ["low_dimensional_.zip","smallcoeff_pisinger.zip","largecoeff_pisinger.zip","hardinstances_pisinger.zip"]
    for file in file_tester:
        zf = zipfile.ZipFile(file)
        name_files = zf.namelist()
        experiments = []
        time = []
        for filename in name_files:
            d = knapsack_instance(zf,filename)
            B = np.random.random((10, d)) < .5
            profits = knapsack_discard(B)
            experiments.append(profits)
            time.append(timeit())

        if "low" in file:
            optimum = np.loadtxt("low_dimensional_optimum.txt", dtype=float)
        else:
            optimum = np.zeros(len(experiments))
            # global kp_data
            # optimum.append(kp_data["optimum"])
            # comp_time.append(kp_data["comp_time"])
        print("Results of ",str(len(name_files)), file, "problems")
        result=pd.DataFrame({'name':name_files,'Result':experiments,'Optimum':optimum,'Difference in % of optimum':(experiments-optimum)*100/optimum,'Time':time})
        print(result)
        result=[]
#...
