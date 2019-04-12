## Python 2.7 compatibility ##
from __future__ import print_function
import matplotlib
matplotlib.use('TkAgg')

## Needed libraries ##
import numpy as np
import time
import matplotlib.pyplot as plt
from pupi_bm import *

## PUPI algorithm: Real-valued (continuous) problems class implementation  ##
class PupiReal():

    ## Initialization of class parameters ##
    def __init__(self, fcost=sphere, LB=np.array([-5.,-5.]), UB=np.array([5.,5.]), \
                 n=20, nw=.25, alpha=0.01, sigma=0.1, max_eval=40000, mode='clipped', viz=False, stats=False):
        self.fcost = fcost           # Cost function (problem) to be optimised
        self.LB = LB                 # Array of variable lower bound in each dimension
        self.UB = UB                 # Array of variable upper bound in each dimension
        self.d = len(LB)             # Problem dimensionality
        self.n = n                   # Population size (number of pigeons)
        self.nw = nw                 # Rate of walkers pigeons in the population
        self.alpha = alpha           # Step size for followers move
        self.sigma = sigma           # Step size for walkers move
        self.max_eval = max_eval     # Max number of cost function evaluations allowed
        self.mode = mode             # Boundary movement mode (clipped or toroid)
        self.viz = viz               # Plot pigeons movements flag (only for 2D problems)
        self.xbest = np.zeros(self.d)# The best solution found
        self.fbest = np.Inf          # The optimal cost of best solution found
        self.ibest = np.Inf          # The iteration were best was found
        self.toc = 0                 # Timing counter
        self.stats = stats           # Record solution statistics per iteration flag
        self.fmins = []; self.favgs = []; self.fmaxs = [] # Holders for solution statistics per iteration

    ## Visualisation setup ##
    def vizSetup(self):
        if self.d==2:
            X = np.linspace(self.LB[0], self.UB[0], 100)
            Y = np.linspace(self.LB[1], self.UB[1], 100)
            self.X, self.Y = np.meshgrid(X, Y)
            self.Z = self.fcost(np.vstack([self.X.flatten(), self.Y.flatten()]).T).reshape(self.X.shape)

    ## Visualise one iteration of optimisation algorithm ##
    def vizIteration(self, i, P, followers, walkers, leader):
        if self.d == 2:
            plt.contourf(self.X, self.Y, self.Z, 8, colors=('navy', 'royalblue', 'skyblue', 'greenyellow', 'yellow', 'darkorange', 'tomato', 'crimson', 'maroon'))
            plt.title("Problem: %s / Evaluations: %d / Best cost so far: %.6f  " % (self.fcost.__name__, i, self.fbest))
            plt.scatter(P[followers, 0], P[followers, 1], marker='^', c='black')
            plt.scatter(P[walkers, 0], P[walkers, 1], marker='d', c='darkgreen')
            plt.scatter(P[leader, 0], P[leader, 1], marker='o', c='red')
            plt.draw(); plt.pause(0.0000001); plt.clf()

    ## Create a random population ##
    def create(self):
        P = np.zeros((self.n, self.d))
        for j in range(self.d):
            P[:, j] = np.random.uniform(self.LB[j], self.UB[j], self.n).reshape((1, self.n))
        followers = []; walkers = []
        return P, followers, walkers

    ## Move walkers around (clipped or toroid mode) ##
    def walk(self, X):
        if self.mode == 'clipped':
            for i in range(len(X)):
                # X[i, :] = np.maximum(LB, np.minimum(UB, X[i, :] + sigma*np.random.randn(2)))
                X[i, :] = np.clip((X[i, :] + self.sigma * np.random.randn(self.d)), self.LB, self.UB)
        elif self.mode == 'toroid':
            for i in range(len(X)):
                B = (self.UB - self.LB) + .1
                X[i, :] = np.fmod(B + np.fmod((X[i, :] + self.sigma * np.random.randn(self.d)) - self.LB, B), B) + self.LB
        return X

    ## Move followers towards leader ##
    def follow(self, X, Xl, sigma=0.001):
         for i in range(len(X)):
            # X[i, :] = X[i, :] + alpha*(Xl-X[i, :])
            X[i, :] = np.clip(X[i, :] + self.alpha * (Xl - X[i, :]) + sigma * np.random.randn(self.d), self.LB, self.UB)
         return X

    ## Return the leader pigeon. NB: minimises by default (argmin) ##
    def getLeader(self, F):
        return np.argmin(F, axis=0)

    ## Assign pigeons roles, depending on period of food supply ##
    def convert(self, i, T, followers, walkers):
        if i % T == 0:          # Begin explotiation phase
            walkers = np.random.choice(self.n, int(self.n * self.nw), replace=False)
            followers = np.setdiff1d(range(self.n), walkers)
        elif i % T > .8*T:    # Begin exploration phase
            walkers = np.arange(self.n)
            followers = []
        return followers, walkers

    ## Print algorithm results ##
    def summary(self):
        print("\n%s\nProblem: %s \nEllapsed time: %.2fs \nBest cost: %.10f \nBest solution: " \
              % ("-" * 80, self.fcost.__name__, self.toc, self.fbest), self.xbest)

    ## Get algorithm results ##
    def getResults(self):
        return self.fcost.__name__, self.toc, self.fbest, self.ibest, self.xbest

    ## Get algorithm stats ##
    def getStats(self):
        return self.fmins, self.favgs, self.fmaxs

    ## Optimisation algorithm ##
    def optimise(self):
        if self.viz: self.vizSetup()                # Initialise visualisign settings
        T = (self.max_eval/self.n)/4                # Set period of food supply
        self.fbest, self.xbest = np.Inf, np.zeros(self.d)     # Initialise solution variables
        tic = time.time()                           # Start execution timer
        P, followers, walkers = self.create()       # Pigeon initial population
        for i in range(0, self.max_eval, self.n):   # Main loop
            F = self.fcost(P)                       # Compute cost of current solution population
            leader = self.getLeader(F)              # Find leader pigeon
            if F[leader] < self.fbest:              # Trace the best solution so far
                self.fbest, self.xbest, self.ibest = F[leader], np.copy(P[leader]), i
            if self.stats:                          # Trace other statistics for current solution
                self.fmins.append(F[leader]); self.favgs.append(np.mean(F)); self.fmaxs.append(F[np.argmax(F, axis=0)])
            followers, walkers = self.convert(i/self.n, T, followers, walkers)  # Set pigeon roles
            P[followers] = self.follow(P[followers], P[leader])          # Move follower pigeons
            P[walkers] = self.walk(P[walkers])                           # Move walker pigeons
            if self.viz and not (i/self.n % 20):    # If visualisation, do it every 20 iterations
                self.vizIteration(i, P, followers, walkers, leader)
        self.toc = time.time() - tic
        return self.fcost.__name__, self.fbest, self.ibest, self.xbest, self.toc

## End of class ##
