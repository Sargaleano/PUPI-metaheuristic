## Python 2.7 compatibility ##
from __future__ import print_function
import matplotlib
import math
matplotlib.use('TkAgg')

## Needed libraries ##
import numpy as np
import time
import matplotlib.pyplot as plt
from pupi_bm import *

'''
PUPI algorithm is a method for unconstrained cost function optimisation 
inspired in the foraging behaviour of urban park pigeons.
(c) 2020 - v2.0
License: CC BY-SA 4.0 
'''

## PUPI algorithm: Real-valued (continuous) problems class implementation  ##
class PupiReal():

    ## Initialization of class parameters ##
    def __init__(self, fcost=sphere, LB=np.array([-5.,-5.]), UB=np.array([5.,5.]), \
                 n=40, nw=.25, alpha=10.0, sigma=0.1, max_eval=40000, mode='clipped', viz=False, stats=True):
        self.fcost = fcost           # Cost function (problem) to be optimised
        self.LB = LB                 # Array of variable lower bound in each dimension
        self.UB = UB                 # Array of variable upper bound in each dimension
        self.d = len(LB)             # Problem dimensionality
        self.n = n                   # Population size (number of pigeons)
        self.nw = nw                 # Rate of walkers pigeons in the population
        self.max_eval = max_eval     # Max number of cost function evaluations allowed
        self.famine = .2*max_eval    # Max number of evaluations with no fitness improvement (famine trigger)
        self.alpha = alpha           # Step size for followers move
        self.sigma = sigma           # Step size for walkers move
        self.mode = mode             # Boundary movement mode (clipped or toroid)
        self.viz = viz               # Plot pigeons movements flag (only for 2D problems)
        self.xbest = np.zeros(self.d)# The best solution found
        self.fbest = np.Inf          # The cost of best solution found
        self.ibest = np.Inf          # The iteration were best was found
        self.toc = 0                 # Timing counter
        self.stats = stats           # Record solution statistics per iteration flag
        self.fmins = []; self.favgs = []; self.fmaxs = [] # Holders for solution statistics per iteration

    ## Core function: Assign pigeons roles, depending on availability of food supply (starvation) ##
    def setRoles(self, starvation=False):
        if not starvation:    # Flocking phase (split flock into followers+walkers)
            walkers = np.random.choice(self.n, int(self.n * self.nw), replace=False)
            followers = np.setdiff1d(range(self.n), walkers)
        else:                 # Stagnation phase (disband, let walkers only)
            walkers, followers = np.arange(self.n), []
        return followers, walkers

    ## Core function: Create a random population ##
    def create(self):
        P = np.zeros((self.n, self.d))
        for j in range(self.d):
            P[:, j] = np.random.uniform(self.LB[j], self.UB[j], self.n).reshape((1, self.n))
        followers, walkers = self.setRoles(starvation=False)
        return P, followers, walkers

    ## Core function: Follower move (bring followers towards leader) ##
    def follow(self, X, Xl, sigma=0.001):
         dispersion = 0
         for i in range(len(X)):                # len(X) is the number of follower pigeons
            j = np.random.choice(self.d, 1)[0]  # Choose a coordinate to update
            delta_x = Xl[j] - X[i, j]
            X[i, j] = np.clip(X[i, j] + self.alpha * (delta_x) + sigma * np.random.randn(), self.LB[j], self.UB[j])
            dispersion += abs(delta_x)
         return X, dispersion

    ## Core function: Walkers move (wander walkers around, either in clipped or toroid mode) ##
    def walk(self, X):
        if self.mode == 'clipped':
            for i in range(len(X)):
                j = np.random.choice(self.d, 1)[0]  # Choose a coordinate to update
                X[i, j] = np.clip((X[i, j] + self.sigma * np.random.randn()), self.LB[j], self.UB[j])
        elif self.mode == 'toroid':
            for i in range(len(X)):
                j = np.random.choice(self.d, 1)[0]  # Choose a coordinate to update
                B = (self.UB[j] - self.LB[j]) + .1
                X[i, j] = np.fmod(B + np.fmod((X[i, j] + self.sigma*(2*np.random.rand()-1))-self.LB[j], B), B) + self.LB[j]
        return X

    ## Core function: Starvation move (disband the flock away with Levy flights) ##
    def disband(self, X):
        if self.mode == 'clipped':
            for i in range(len(X)):
                j = np.random.choice(self.d, 1)[0]  # Choose a coordinate to update
                X[i, j] = np.clip((X[i, j] + self.sigma * self.randLevy(1.9)), self.LB[j], self.UB[j])
                # X[i, j] = np.clip((X[i, j] + self.sigma * self.randLevy(1.2)), self.LB[j], self.UB[j])
        elif self.mode == 'toroid':
            for i in range(len(X)):
                j = np.random.choice(self.d, 1)[0]  # Choose a coordinate to update
                B = (self.UB[j] - self.LB[j]) + .1
                X[i, j] = np.fmod(B + np.fmod((X[i, j] + self.sigma*(2*self.randLevy(1.9)-1)) - self.LB[j], B), B) + self.LB[j]
        return X

    ## Core function: Update the value of the parameter alpha (followers step) ##
    # def updateAlpha(self, P, Ppast):
        # delta_x = 0
        # for i in range(len(P)):
        #     delta_x += np.linalg.norm(np.subtract(P[i, :], Ppast[i, :]))
        # delta_x = delta_x / len(P)
        # if delta_x < 0.0001:
        #     self.alpha = 0.00000000001
        # if delta_x >= 0.0001 and delta_x < 0.01:
        #     self.alpha = 0.00001
        # if delta_x >= 0.01:
        #     self.alpha = 0.1
        # self.alpha = np.exp(delta_x/self.UB[0]) - 1
        # self.alpha = (0.01 ** (1/delta_x))
        # self.alpha = 0.485 + 0.119*math.log(delta_x)
        # return self.alpha

    ## Core function: Update the value of the parameter alpha (followers step) ##
    # def updateAlpha1(self, P):
        ## dispersion = np.average(np.var(P, axis=1))
        # dispersion = np.sum(np.var(P, axis=1))
        # if dispersion < 0.000001:
        #     self.alpha = (1-np.exp(-dispersion))
        # else:
        #     self.alpha = 0.1
        # if dispersion >= 0.0001 and dispersion < 0.01:
        #     self.alpha = 0.00001
        # if dispersion >= 0.01:
        #     self.alpha = 0.1
        # self.alpha = np.exp(dispersion/self.UB[0]) - 1
        # self.alpha = (0.01 ** (1/dispersion))
        # self.alpha = 0.485 + 0.119*math.log(dispersion)
        # return dispersion

    ## Core function: Update the value of the parameter alpha (followers step) ##
    def updateAlpha(self, dispersion):
        # execute = False
        if dispersion < (self.UB[0]*0.5):
            self.alpha = (1 - np.exp(-dispersion))
            # execute = True
        else:
            self.alpha = 0.1
        # return execute

    ## Core function: Return the leader pigeon. NB: minimises by default (argmin) ##
    def getLeader(self, F):
        return np.argmin(F, axis=0)

    ## Core function: Genotype-phenotype mapping function, when different to direct mapping ##
    def gpm(self, P):
        return P                        # Direct mapping: Phenotype is the same genotype

    ## Core function: Print algorithm results ##
    def summary(self):
        # print("\n%s\nProblem: %s (d=%d)\nEllapsed time: %.2fs \nBest cost: %.5f \nFound after: %d evaluations " \
        #       % ("-" * 80, self.fcost.__name__, self.d, self.toc, self.fbest, self.ibest))
        # print("Best solution (genotype): ", list(map(float, ["%.3f" % v for v in self.xbest])))
        # print("Best solution (phenotype): ", self.gpm(self.xbest))
        return self.toc, self.fbest

    ## Get algorithm results ##
    def getResults(self):
        return self.fcost.__name__, self.d, self.fbest, self.ibest, self.xbest, self.toc, self.n, self.nw, self.alpha, self.sigma, self.max_eval, self.mode

    ## Core function: Optimisation (search) algorithm ##
    def optimise(self):
        if self.viz: self.vizSetup()                # Initialise visualisation settings
        self.fbest, self.xbest = np.Inf, np.zeros(self.d)     # Set initial solution
        tic = time.time()                           # Start execution timer
        hunger, starvation = 0, False               # Set foraging/starvation behaviour parameters
        P, followers, walkers = self.create()       # Create pigeon initial population
        # alpha_hist = np.array([self.alpha])
        # dispersion_hist = np.array([self.alpha])
        # deltax_hist = np.zeros(0)
        #times = np.empty(1)  # Array with each iteration time
        counter = 0

        for i in range(0, self.max_eval, self.n):   # Main search loop, iterates every n evaluations
            # tic_iter = time.time()                  # Start execution timer for each
            F = self.fcost(self.gpm(P))             # Compute cost of current pigeon population
            leader = self.getLeader(F)              # Find leader pigeon
            # Ppast = np.copy(P[followers])                   # Copy initial followers position
            if F[leader] < self.fbest:              # Update best solution found (minimisation)
                self.fbest, self.xbest, self.ibest = F[leader], np.copy(P[leader]), i
                hunger = 0                            # Since better solution found, reset idleness counter
            if not starvation:
                P[followers], dispersion = self.follow(P[followers], P[leader])              # Move followers
                P[walkers] = self.walk(P[walkers])                               # Move walkers
                P[0] = np.copy(self.xbest); P[0] = self.walk(P[[0]])             # Enforce elitism
                # deltax_hist = np.append(deltax_hist, self.updateAlpha(P[followers], Ppast))
                # self.updateAlpha(P[followers], Ppast)
                # dispersion_hist = np.append(dispersion_hist, self.updateAlpha1(P[followers]))
                # self.updateAlpha1(P[followers])
                self.updateAlpha(dispersion)
                # execution = self.updateAlpha(dispersion)
                # if execution:
                #     counter +=1
                # alpha_hist = np.append(alpha_hist, self.alpha)
                hunger += self.n                                                 # Increase flock hungriness
                starvation = (hunger >= self.famine)                             # Check if flock is starving
                if starvation:
                    followers, walkers = self.setRoles(starvation=True)          # Start starvation (disband) phase
            else:
                P[walkers] = self.disband(P[walkers])                            # Disband pigeons
                hunger -= self.n                                                 # Decrease flock hungriness
                starvation = (hunger > 0)                                        # End starvation when no hungry anymore
                if not starvation:
                    followers, walkers = self.setRoles(starvation=False)         # Restore foraging phase
            ## Statistics collection and visualisation ##
            if self.stats:
                self.fmins.append(F[leader]); self.favgs.append(np.mean(F)); self.fmaxs.append(F[np.argmax(F, axis=0)])
            if self.viz and not (i / self.n % 10):  # If visualisation, do it every 10 iterations
                self.vizIteration(i, P, followers, walkers, leader)
            # toc_iter = time.time() - tic_iter
            # times = np.append(times, toc_iter)
        self.toc = time.time() - tic                # Stop and record execution timer
        # return counter

        # plt.plot(alpha_hist)
        # plt.xlabel("Iteraciones")
        # plt.ylabel("Alpha")
        # plt.title("Valor del parámetro alpha por iteración Problema: %s / Max Eval: %d / Tamaño población: %d" % (
        # self.fcost.__name__, self.max_eval, self.n))
        # plt.show()
        # plt.plot(dispersion_hist)
        # plt.xlabel("Iteraciones")
        # plt.ylabel("Dispersion")
        # plt.title("Dispersión de las palomas followers por iteración Problema: %s / Max Eval: %d / Tamaño población: %d" % (
        #     self.fcost.__name__, self.max_eval, self.n))
        # plt.show()
        # # plt.plot(deltax_hist)
        # plt.plot(times[1:])
        # plt.xlabel("Iteraciones")
        # plt.ylabel("Tiempo")
        # plt.title("Tiempo transcurrido por iteración Problema: %s / Max Eval: %d / Tamaño población: %d" % (self.fcost.__name__, self.max_eval, self.n))
        # plt.show()

    ## Ancillary function: Generate a Levy distributed variate. Note: 1 < beta < 2 ##
    def randLevy(self, beta):
        num = np.random.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        den = np.random.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
        std = (num / den) ** (1 / beta)
        u = np.random.normal(0, std * std)
        v = np.random.normal()
        return u / (abs(v) ** (1 / beta))

    ## Ancillary function: Get algorithm results ##
    def getResults(self):
        return self.fcost.__name__, self.d, self.fbest, self.ibest, self.xbest, \
               self.toc, self.n, self.nw, self.alpha, self.sigma, self.max_eval, self.mode

    ## Ancillary function: Get algorithm stats ##
    def getStats(self):
        return self.fmins, self.favgs, self.fmaxs

    ## Ancillary function: Visualisation setup ##
    def vizSetup(self):
        X = np.linspace(self.LB[0], self.UB[0], 100)
        Y = np.linspace(self.LB[1], self.UB[1], 100)
        self.X, self.Y = np.meshgrid(X, Y)
        self.Z = self.fcost(np.vstack([self.X.flatten(), self.Y.flatten()]).T).reshape(self.X.shape)

    ## Ancillary function: Visualise one iteration of optimisation algorithm ##
    def vizIteration(self, i, P, followers, walkers, leader):
        plt.contourf(self.X, self.Y, self.Z, 8, colors=('navy', 'royalblue', 'skyblue', 'greenyellow', 'yellow', 'darkorange', 'tomato', 'crimson', 'maroon'))
        plt.title("Problem: %s / Evaluations: %d / Best cost so far: %.6f  " % (self.fcost.__name__, i, self.fbest))
        plt.scatter(P[followers, 0], P[followers, 1], marker='^', c='black')
        plt.scatter(P[walkers, 0], P[walkers, 1], marker='d', c='darkgreen')
        plt.scatter(P[leader, 0], P[leader, 1], marker='o', c='red')
        plt.scatter(self.xbest[0], self.xbest[1], marker='*', c='white')
        plt.xlim(self.LB[0], self.UB[0]); plt.ylim(self.LB[1], self.UB[1])
        plt.draw(); plt.pause(0.0000001); plt.clf()

## End of class ##


## PUPI algorithm: Binary-valued problems class implementation  ##
class PupiBinary(PupiReal):

    ## Initialization of class parameters ##
    def __init__(self, fcost=oneMax, d=64, n=20, nw=0.25, alpha=0.1, sigma=1, max_eval=40000, viz=True, stats=True):
        # Set parameters in super-class, LB and UB are constrained to unit-interval #
        PupiReal.__init__(self, fcost=fcost, LB=np.zeros(d), UB=np.ones(d), n=n, nw=nw, \
                          alpha=alpha, sigma=sigma, max_eval=max_eval, mode='toroid', viz=viz, stats=stats)

    ## Core function: Genotype-phenotype mapping function: Maps real unit-interval to binary values ##
    def gpm(self, P):
        return (P >= .5)*1     # Apply threshold and cast True/False values to 1/0

    ## Ancillary function: Get algorithm results ##
    def getResults(self):
        return self.fcost.__name__, self.d, self.fbest, self.ibest, self.gpm(self.xbest), \
               self.toc, self.n, self.nw, self.alpha, self.sigma, self.max_eval, self.mode

    ## Ancillary function: Visualise one iteration of optimisation algorithm ##
    def vizIteration(self, i, P, followers, walkers, leader):
        if self.d == 2:                         # If 2D, plot pigeons on surface
            PupiReal.vizIteration(self, i, P, followers, walkers, leader)
        elif self.d in np.arange(3, 11)**2:     # If perfect-square dimensions, plot solution as bitmap
            m = int(np.sqrt(self.d))
            plt.imshow(self.gpm(self.xbest).reshape(m, m), cmap=('PuBu'), vmin=0, vmax=1)
            plt.title("Problem: %s / Evaluations: %d / Best cost so far: %.2f  " % (self.fcost.__name__, i, self.fbest))
            plt.draw(); plt.pause(0.0000001); plt.clf()

## End of class ##

## PUPI Ensemble: Outputs the average of weak binary-valued PUPIs  ##
class PupiEnsemble(PupiBinary):

    ## Core function: Optimisation (search) algorithm ##
    def optimise(self):
        tic = time.time()                           # Start overall execution timer
        xweak = np.zeros(self.d)
        nweak = 11
        for _ in range(nweak):
            PupiBinary.optimise(self)
            xweak = xweak+self.gpm(self.xbest)
            print(xweak)
        self.xbest = xweak/nweak
        self.fbest = self.fcost(np.array([self.gpm(self.xbest)]))[0]
        self.toc = time.time() - tic                # Stop overall execution timer

## End of class ##
