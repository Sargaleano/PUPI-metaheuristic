## Import libraries ##
from pupi_class import PupiReal, PupiBinary
from pupi_bm import *
import numpy as np
import time
import pandas as pd
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

    ## Open file .zip to tester##

    file_tester = ["low_dimensional_.zip","smallcoeff_pisinger.zip","largecoeff_pisinger.zip","hardinstances_pisinger.zip"]

    ## Read file´s name in .zip##
    for file in file_tester:
        zf = zipfile.ZipFile(file) #abrir el .zip
        name_files = zf.namelist() #almacenar los nombres de los archivos en el .zip
        experiments = []  #almacena los resultado del ejercicio para cada archivo

        ##para cada archivo ejecuta el ejercicio##
        for filename in name_files:
            d = knapsack_instance(zf,filename)[0]
            v_optimum=knapsack_instance(zf,filename)[1]
            time_optimum = knapsack_instance(zf, filename)[2]
            B = np.random.random((10, d)) < .5
            t0=time.time()
            profits_penalty = knapsack_penalty(B)
            t_penalty = time.time() - t0
            t0=time.time()
            profits_discard = knapsack_discard(B)
            t_discard=time.time()-t0
            experiments.append([filename,profits_discard,profits_penalty, v_optimum, t_discard,t_penalty,time_optimum])

        print("Results of ",str(len(name_files)), file, "problems")
        result=pd.DataFrame(experiments, columns=['filename','profits_discard','profits_penalty', 'v_optimum', 't_discard','t_penalty','time_optimum'])
        if 'low' in file:
            result['v_optimum']=np.loadtxt("low_dimensional_optimum.txt")
            #result['difference_value']=(result['v_optimum'] - result['profits'])
        pd.options.display.max_columns = None
        print(result)
        experiments=[]
#...
