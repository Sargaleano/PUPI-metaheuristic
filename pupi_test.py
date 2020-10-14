'''
A test script for solving different optimisation tasks with PUPI
PUPI algorithm is a method for unconstrained cost function optimisation
inspired in the foraging behaviour of urban park pigeons.
(c) 2020 - v3.0 (October/2020)
License: GPLv3 (see https://www.gnu.org/licenses/gpl-3.0.txt)
'''

## Import libraries ##
from pupi_class import PupiReal, PupiBinary, PupiEnsemble
from pupi_bm import *
import numpy as np
import time
import matplotlib.pyplot as plt

np.random.seed(int(str(int(time.time() * 1000))[-8:-1]))  # Set random generator seed
# viz = True        # Visualisation on
viz = False         # Visualisation off
nreps = 10          # Set number of experiment repetitions

if(True):
    #### Continuous-valued experiments ####

    # 2D problems #
    problems = []#rastrigin, rastrigin_offset, rastrigin_bipolar, rosenbrock, himmelblau, sphere]
    for _ in range(nreps):
        for problem in problems:
            pupic = PupiReal(fcost=problem, viz=viz)
            pupic.optimise()
            pupic.summary()

    # Eggholder 2D problem has different bounds and therefore step sizes #
        pupic = PupiReal(fcost=eggholder, LB=np.array([-512., -512.]), UB=np.array([512., 512.]), nw=.25, alpha=.9, sigma=50, viz=viz, mode='clipped')
        pupic.optimise()
        pupic.summary()
    # quit(1)  # Uncomment to run only above exps

    # 10D problems #
    d = 10
    UB = np.repeat(5., d); LB = -UB
    problems = [sphere, rastrigin, rastrigin_offset, rastrigin_bipolar]
    for problem in problems:
        pupic = PupiReal(fcost=problem, LB=LB, UB=UB, max_eval=100000, n=20, nw=.1, alpha=.1, sigma=1)
        pupic.optimise()
        pupic.summary()
    # quit(1)  # Uncomment to run only above exps

    # 30D problems #
    d = 30
    UB = np.repeat(5., d); LB = -UB
    problems = [sphere, rastrigin, rastrigin_offset, rastrigin_bipolar]
    for problem in problems:
        pupic = PupiReal(fcost=problem, LB=LB, UB=UB, max_eval=300000, n=20, nw=.1, alpha=.1, sigma=1)
        pupic.optimise()
        pupic.summary()


if(True):
    #### Binary-valued experiments ####

    problems = [oneMax, squareWave, powSum]#, binVal]

    for _ in range(nreps):
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

        # 1000D problems #
        for problem in problems:
            pupib = PupiEnsemble(fcost=problem, d=1000, n=10, nw=.1, alpha=.5, sigma=1, max_eval=50000)
            pupib.optimise()
            pupib.summary()


if(True):
    #### Combinatorial experiments ####

    # kp = knapsack_instance("f4_l-d_kp_4_11")
    kp = knapsack_instance("f10_l-d_kp_20_879")

    problems = [knapsack_discard]

    for _ in range(nreps):
        for problem in problems:
            pupib = PupiBinary(fcost=problem, d=kp["n_items"], n=40, nw=.1, alpha=.5, sigma=1, max_eval=40000, viz=viz)
            pupib.optimise()
            pupib.summary()
