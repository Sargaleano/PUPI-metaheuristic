## Python 2.7 compatibility ##
from __future__ import print_function
import matplotlib
matplotlib.use('TkAgg')

## Needed libraries ##
import numpy as np
import time
import matplotlib.pyplot as plt

## Benchmark definitions (numpy implementation) ##
def sphere(X):
    return np.sum(X**2, axis=1)

def rosenbrock(X):
    return (1 - X[:,0])**2 + 100 * (X[:,1] - (X[:,0]**2))**2

def rastrigin(X):
    d = X.shape[1]      # Space dimensionality
    return np.sum((X**2 - 10*np.cos(2*np.pi*X)), axis=1) + 10*d

def himmelblau(X):
    return (X[:,0]**2 + X[:,1] - 11)**2 + (X[:,0] + X[:,1]**2 - 7)**2

def eggholder(X):
    Z = X[:,1]+47
    return (-Z * np.sin(np.sqrt((np.abs(X[:,0]/2 + Z)))) \
            -X[:,0] * np.sin(np.sqrt((np.abs(X[:,0] - Z))))) #+ 959.640662720851

## Create a random population ##
# Inputs: LowerBound array, UpperBound array, pop size
def create(LB, UB, n):
    P = np.zeros((n, len(LB)))
    for j in range(len(LB)):
        P[:, j] = np.random.uniform(LB[j], UB[j], n).reshape((1, n))
    return P

## Move walkers around (clipped or toroidal mode) ##
def walk(X, sigma, LB, UB, mode='clipped'):
    n, d = X.shape
    if mode=='clipped':
        for i in range(n):
            #X[i, :] = np.maximum(LB, np.minimum(UB, X[i, :] + sigma*np.random.randn(2)))
            X[i, :] = np.clip((X[i, :] + sigma*np.random.randn(d)), LB, UB)
    elif mode=='toroid':
        for i in range(n):
            B = (UB - LB) + .1
            X[i, :] = np.fmod(B + np.fmod((X[i, :] + sigma * np.random.randn(d)) - LB, B), B) + LB
    return X

## Move followers towards leader ##
def follow(X, alpha, Xl, LB, UB, sigma=0.001):
    n, d = X.shape
    for i in range(n):
        #X[i, :] = X[i, :] + alpha*(Xl-X[i, :])
        X[i, :] = np.clip(X[i, :] + alpha*(Xl - X[i, :]) + sigma*np.random.randn(d), LB, UB)
    return X

## Return the leader pigeon ##
# Inputs: cost function array #
# NB: minimises by default (argmin) #
def get_leader(F, P):
    return np.argmin(F, axis = 0)

def optimise(n, nw, max_iter, alpha, sigma, fcost, LB, UB, viz=False):
    if viz:
        ## Visialization variables ##
        X = np.linspace(LB[0], UB[0], 100)
        Y = np.linspace(LB[1], UB[1], 100)
        X, Y = np.meshgrid(X, Y)
        Z = fcost(np.vstack([X.flatten(), Y.flatten()]).T).reshape(X.shape)

## PUPI algorithm ##
    #np.random.seed(19680801)       # Fixing random state for reproducibility
    tic = time.time()
    T = max_iter/4                  # Period of food supply
    P = create(LB, UB, n)           # Pigeon population
    fbest, xbest = np.Inf, np.zeros(len(LB))
    for i in range(max_iter):
        F = fcost(P) 
        leader = get_leader(F, P)
        if F[leader] < fbest:       # Keep track the best solution so far
            fbest, xbest = F[leader], np.copy(P[leader])
        if i % T == 0:              # Convert pigeons role
            walkers = np.random.choice(n, int(n * nw), replace=False)
            followers = np.setdiff1d(range(n), walkers)
        elif i % T > .8*T:
            walkers = np.arange(n)
            followers = []
        P[followers] = follow(P[followers], alpha, P[leader], LB, UB)
        P[walkers] = walk(P[walkers], sigma, LB, UB)

        if viz and not (i%20):      # If visualisation, do it every 20 iterations
            plt.contourf(X, Y, Z, 8, colors=('navy', 'royalblue', 'skyblue', 'greenyellow', 'yellow', 'darkorange', 'tomato', 'crimson', 'maroon'))
            plt.title("Problem: %s / Iteration: %d / Best cost so far: %.10f  " % (fcost.__name__, i, fbest))
            plt.scatter(P[followers,0], P[followers,1], marker='^', c='black')
            plt.scatter(P[walkers,0], P[walkers,1], marker='d', c='darkgreen')
            plt.scatter(P[leader,0], P[leader,1], marker='o', c='red')
            plt.draw()
            plt.pause(0.0000001)
            plt.clf()
    toc = time.time() - tic
    print("\n%s\nProblem: %s \nEllapsed time: %.2fs \nBest cost: %.10f \nBest solution/Leader: " \
          % ("-"*80, fcost.__name__, toc, fbest), xbest, P[leader])
    return fcost.__name__, toc, fbest, xbest


