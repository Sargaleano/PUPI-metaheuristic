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
                 n=20, nw=.25, alpha=0.01, sigma=0.1, max_eval=40000, mode='clipped', viz=True, stats=False):
        self.fcost = fcost           # Cost function (problem) to be optimised
        self.LB = LB                 # Array of variable lower bound in each dimension
        self.UB = UB                 # Array of variable upper bound in each dimension
        self.d = len(LB)             # Problem dimensionality
        self.n = n                   # Population size (number of pigeons)
        self.nw = nw                 # Rate of walkers pigeons in the population
        self.max_eval = max_eval     # Max number of cost function evaluations allowed
        self.T = (max_eval/n)/10     # Period of food supply before exhaustion
        self.alpha = alpha           # Step size for followers move
        self.sigma = sigma           # Step size for walkers move
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
                B = (self.UB - self.LB) + .01
        #        X[i, :] = np.fmod(B + np.fmod((X[i, :] + self.sigma * np.random.randn(self.d)) - self.LB, B), B) + self.LB
                X[i, :] = np.fmod(B + np.fmod((X[i, :] + self.sigma * (2*np.random.rand(self.d)-1)) - self.LB, B), B) + self.LB
        return X

    ## Move followers towards leader ##
    def follow(self, X, Xl, sigma=0.01):#sigma=0.001):
         for i in range(len(X)):
            # X[i, :] = X[i, :] + alpha*(Xl-X[i, :])
            X[i, :] = np.clip(X[i, :] + self.alpha * (Xl - X[i, :]) + sigma * np.random.randn(self.d), self.LB, self.UB)
         return X

    ## Return the leader pigeon. NB: minimises by default (argmin) ##
    def getLeader(self, F):
        return np.argmin(F, axis=0)

    ## Assign pigeons roles, depending on period of food supply ##
    def convert(self, i, followers, walkers):
        if i % self.T == 0:             # Begin explotiation phase
            walkers = np.random.choice(self.n, int(self.n * self.nw), replace=False)
            followers = np.setdiff1d(range(self.n), walkers)
        elif i % self.T > .9*self.T:    # Begin exploration phase
            walkers = np.arange(self.n)
            followers = []
        return followers, walkers

    ## Genotype-phenotype mapping function, when direct mapping not applicable ##
    def gpm(self, P):
        return P                        # Direct mapping: Phenotype is the same genotype

    ## Print algorithm results ##
    def summary(self):
        # print("\n%s\nProblem: %s \nEllapsed time: %.2fs \nBest cost: %.10f \nBest solution (genotype): " \
        #       % ("-" * 80, self.fcost.__name__, self.toc, self.fbest), map(float, ["%.2f" % v for v in self.xbest]), self.gpm(self.xbest))
        print("\n%s\nProblem: %s (d=%d)\nEllapsed time: %.2fs \nBest cost: %.2f " % ("-" * 80, self.fcost.__name__, self.d, self.toc, self.fbest))
        print("Best solution (genotype): ", list(map(float, ["%.3f" % v for v in self.xbest])))
        print("Best solution (phenotype): ", self.gpm(self.xbest))

    ## Get algorithm results ##
    def getResults(self):
        return self.fcost.__name__, self.d, self.fbest, self.ibest, self.xbest, self.toc, self.n, self.alpha, self.sigma

    ## Get algorithm stats ##
    def getStats(self):
        return self.fmins, self.favgs, self.fmaxs

    ## Optimisation algorithm ##
    def optimise(self):
        if self.viz: self.vizSetup()                # Initialise visualisation settings
        self.fbest, self.xbest = np.Inf, np.zeros(self.d)     # Initialise solution variables
        tic = time.time()                           # Start execution timer
        P, followers, walkers = self.create()       # Pigeon initial population
        for i in range(0, self.max_eval, self.n):   # Main loop
            F = self.fcost(self.gpm(P))             # Compute cost of current solution population
            leader = self.getLeader(F)              # Find leader pigeon
            if F[leader] < self.fbest:              # Update best solution found
                self.fbest, self.xbest, self.ibest = F[leader], np.copy(P[leader]), i
            followers, walkers = self.convert(i/self.n, followers, walkers)  # Set pigeon roles
            P[followers] = self.follow(P[followers], P[leader])              # Move follower pigeons
            P[walkers] = self.walk(P[walkers])                               # Move walker pigeons

            ## Statistics collection and visualisation ##
            if self.stats:
                self.fmins.append(F[leader]); self.favgs.append(np.mean(F)); self.fmaxs.append(F[np.argmax(F, axis=0)])
            if self.viz and not (i / self.n % 20):  # If visualisation, do it every 20 iterations
                self.vizIteration(i, P, followers, walkers, leader)
        self.toc = time.time() - tic

## End of class ##


## PUPI algorithm: Binary-valued problems class implementation  ##
class PupiBinary(PupiReal):

    ## Initialization of class parameters ##
    def __init__(self, fcost=oneMax, d=64, n=30, nw=0.1, alpha=0.1, sigma=0.1, max_eval=40000, viz=True, stats=True):
        # Set parameters in super-class, LB and UB are constrained to unit-interval #
        PupiReal.__init__(self, fcost=fcost, LB=np.zeros(d), UB=np.ones(d), \
                          n=n, nw=nw, alpha=alpha, sigma=sigma, max_eval=max_eval,  \
                          mode='toroid', viz=viz, stats=stats)
        self.T = (self.max_eval/self.n)/20          # Decrease period of food supply before exhaustion

    ## Genotype-phenotype mapping function: Maps real unit-interval to binary values ##
    def gpm(self, P):
        return (P >= .5)*1     # Apply threshold and cast True/False values to 1/0

    ## Get algorithm results ##
    def getResults(self):
        return self.fcost.__name__, self.d, self.fbest, self.ibest, self.gpm(self.xbest), self.toc, self.n, self.alpha, self.sigma, self.max_eval

    ## Visualise one iteration of optimisation algorithm ##
    def vizIteration(self, i, P, followers, walkers, leader):
        if self.d == 2:                         # If 2D, plot pigeons on surface
            PupiReal.vizIteration(self, i, P, followers, walkers, leader)
        elif self.d in np.arange(3, 11)**2:     # If perfect-square dimensions, plot solution as bitmap
            m = int(np.sqrt(self.d))
            plt.imshow(self.gpm(self.xbest).reshape(m, m), cmap=('PuBu'), vmin=0, vmax=1)
            plt.title("Problem: %s / Evaluations: %d / Best cost so far: %.2f  " % (self.fcost.__name__, i, self.fbest))

            # plt.rcParams.update({'font.size': 20})
            # plt.xticks([0,1,2,3,4,5,6,7]); plt.yticks([0,1,2,3,4,5,6,7])
            # plt.rcParams.update({'savefig.bbox': 'tight'})
            # if i in [600, 6000, 18000, 39600]: print("Saving..."); plt.savefig("results/bmp-%s-%d.png" % (self.fcost.__name__, i), dpi=300)

            plt.draw(); plt.pause(0.0000001); plt.clf()

## End of class ##