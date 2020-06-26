## Needed libraries ##
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import time
import matplotlib.pyplot as plt
from pupi_bm import *

## PUPI algorithm: Real-valued (continuous) problems class implementation  ##
class PupiReal():

    ## Initialization of class parameters ##
    def __init__(self, fcost=sphere, LB=np.array([-5.,-5.]), UB=np.array([5.,5.]), \
                 n=40, nw=.25, alpha=0.01, sigma=0.1, max_eval=40000, mode='clipped', viz=False, stats=False):
        self.fcost = fcost           # Cost function (problem) to be optimised
        self.LB = LB                 # Array of variable lower bound in each dimension
        self.UB = UB                 # Array of variable upper bound in each dimension
        self.d = len(LB)             # Problem dimensionality
        self.n = n                   # Population size (number of pigeons)
        self.nw = nw                 # Rate of walkers pigeons in the population
        self.max_eval = max_eval     # Max number of cost function evaluations allowed
        self.T = (max_eval/n)/10     # Period of food supply duration
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

    ## Visualisation setup ##
    def vizSetup(self):
        X = np.linspace(self.LB[0], self.UB[0], 100)
        Y = np.linspace(self.LB[1], self.UB[1], 100)
        self.X, self.Y = np.meshgrid(X, Y)
        self.Z = self.fcost(np.vstack([self.X.flatten(), self.Y.flatten()]).T).reshape(self.X.shape)

    ## Visualise one iteration of optimisation algorithm ##
    def vizIteration(self, i, P, followers, walkers, leader):
        plt.contourf(self.X, self.Y, self.Z, 8, colors=('navy', 'royalblue', 'skyblue', 'greenyellow', 'yellow', 'darkorange', 'tomato', 'crimson', 'maroon'))
        plt.title("Problem: %s / Evaluations: %d / Best cost so far: %.6f  " % (self.fcost.__name__, i, self.fbest))
        plt.scatter(P[followers, 0], P[followers, 1], marker='^', c='black')
        plt.scatter(P[walkers, 0], P[walkers, 1], marker='d', c='darkgreen')
        plt.scatter(P[leader, 0], P[leader, 1], marker='o', c='red')
        plt.scatter(self.xbest[0], self.xbest[1], marker='*', c='white')
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
                j = np.random.choice(self.d, 1)[0]  # Choose a coordinate to update (as a scalar)
                X[i, j] = np.clip((X[i, j] + self.sigma * np.random.randn()), self.LB[j], self.UB[j])
        elif self.mode == 'toroid':
            for i in range(len(X)):
                j = np.random.choice(self.d, 1)[0]  # Choose a coordinate to update (as a scalar)
                B = (self.UB[j] - self.LB[j]) + .1
                X[i, j] = np.fmod(B + np.fmod((X[i, j] + self.sigma * (2*np.random.rand()-1)) - self.LB[j], B), B) + self.LB[j]
        return X

    ## Move followers towards leader ##
    def follow(self, X, Xl, sigma=0.001):
         for i in range(len(X)):
             j = np.random.choice(self.d, 1)[0] # Choose a coordinate to update
             X[i, j] = np.clip(X[i, j] + self.alpha * (Xl[j] - X[i, j]) + sigma * np.random.randn(), self.LB[j], self.UB[j])
         return X

    ## Return the leader pigeon. NB: minimises by default (argmin) ##
    def getLeader(self, F):
        return np.argmin(F, axis=0)

    ## Revamp pigeons roles, depending on availability of food supply ##
    def toggle(self, i, followers, walkers):
        if i % self.T == 0:             # Begin explotiation phase
            walkers = np.random.choice(self.n, int(self.n * self.nw), replace=False)
            followers = np.setdiff1d(range(self.n), walkers)
        elif i % (.9 * self.T) == 0:    # Begin exploration phase
            walkers = np.arange(self.n)
            followers = []
        return followers, walkers

    ## Genotype-phenotype mapping function, when different to direct mapping ##
    def gpm(self, P):
        return P                        # Direct mapping: Phenotype is the same genotype

    ## Print algorithm results ##
    def summary(self):
        print("\n%s\nProblem: %s (d=%d)\nEllapsed time: %.2fs \nBest cost: %.5f \nFound after: %d evaluations " \
              % ("-" * 80, self.fcost.__name__, self.d, self.toc, self.fbest, self.ibest))
        print("Best solution (genotype): ", list(map(float, ["%.3f" % v for v in self.xbest])))
        print("Best solution (phenotype): ", self.gpm(self.xbest))

    ## Get algorithm results ##
    def getResults(self):
        return self.fcost.__name__, self.d, self.fbest, self.ibest, self.xbest, \
               self.toc, self.n, self.nw, self.alpha, self.sigma, self.max_eval, self.mode

    ## Get algorithm stats ##
    def getStats(self):
        return self.fmins, self.favgs, self.fmaxs

    ## Optimisation algorithm ##
    def optimise(self):
        if self.viz: self.vizSetup()                # Initialise visualisation settings
        self.fbest, self.xbest = np.Inf, np.zeros(self.d)     # Set initial solution
        tic = time.time()                           # Start execution timer
        P, followers, walkers = self.create()       # Pigeon initial population
        for i in range(0, self.max_eval, self.n):   # Main evolutionary loop
            F = self.fcost(self.gpm(P))             # Compute cost of current solution population
            leader = self.getLeader(F)              # Find leader pigeon
            if F[leader] < self.fbest:              # Update best solution found (minimisation)
                self.fbest, self.xbest, self.ibest = F[leader], np.copy(P[leader]), i
            followers, walkers = self.toggle(i/self.n, followers, walkers)   # Toggle pigeon roles if needed
            P[followers] = self.follow(P[followers], P[leader])              # Move follower pigeons
            P[walkers] = self.walk(P[walkers])                               # Move walker pigeons
            P[0] = np.copy(self.xbest); P[0] = self.walk(P[[0]])             # Enforce elistism

            ## Statistics collection and visualisation ##
            if self.stats:
                self.fmins.append(F[leader]); self.favgs.append(np.mean(F)); self.fmaxs.append(F[np.argmax(F, axis=0)])
            if self.viz and not (i / self.n % 20):  # If visualisation, do it every 20 iterations
                self.vizIteration(i, P, followers, walkers, leader)
        self.toc = time.time() - tic                # Stop execution timer

## End of class ##


## PUPI algorithm: Binary-valued problems class implementation  ##
class PupiBinary(PupiReal):

    ## Initialization of class parameters ##
    def __init__(self, fcost=oneMax, d=64, n=20, nw=0.25, alpha=0.1, sigma=1, max_eval=40000, viz=True, stats=True):
        # Set parameters in super-class, LB and UB are constrained to unit-interval #
        PupiReal.__init__(self, fcost=fcost, LB=np.zeros(d), UB=np.ones(d), \
                          n=n, nw=nw, alpha=alpha, sigma=sigma, max_eval=max_eval,  \
                          mode='clipped', viz=viz, stats=stats)
        self.T = (self.max_eval/self.n)/10          # Decrease period of food supply before exhaustion

    ## Genotype-phenotype mapping function: Maps real unit-interval to binary values ##
    def gpm(self, P):
        return (P >= .5)*1     # Apply threshold and cast True/False values to 1/0

    ## Get algorithm results ##
    def getResults(self):
        return self.fcost.__name__, self.d, self.fbest, self.ibest, self.gpm(self.xbest), \
               self.toc, self.n, self.nw, self.alpha, self.sigma, self.max_eval, self.mode

    ## Visualise one iteration of optimisation algorithm ##
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

    ## Optimisation algorithm ##
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
