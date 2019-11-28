## Import libraries ##
from pupi_class import PupiReal, PupiBinary
from pupi_bm import *
import numpy as np
import time
import matplotlib.pyplot as plt

np.random.seed(int(str(int(time.time() * 1000))[-8:-1]))  # Set random generator seed
viz = True
# viz = False

#### Continuous-valued experiments ####

# 2D problems #
problems = [sphere, rastrigin, rastrigin_offset, rastrigin_bipolar, rosenbrock, himmelblau]
for problem in problems:
    pupic = PupiReal(fcost=problem, viz=viz)
    pupic.optimise()
    pupic.summary()

# Eggholder 2D problem has different bounds and therefore step sizes #
pupic = PupiReal(fcost=eggholder, LB=np.array([-512., -512.]), UB=np.array([512., 512.]), alpha=1, sigma=50, viz=viz)
pupic.optimise()
pupic.summary()

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

problems = [oneMax, squareWave, binVal, powSum]

# 100D problems #
for problem in problems:
    pupib = PupiBinary(fcost=problem, d=100, n=40, nw=.25, alpha=.1, sigma=1, max_eval=50000, viz=viz)
    pupib.optimise()
    pupib.summary()

# 400D problems #
for problem in problems:
    pupib = PupiBinary(fcost=problem, d=400, n=20, nw=.1, alpha=.1, sigma=1, max_eval=100000)
    pupib.optimise()
    pupib.summary()


