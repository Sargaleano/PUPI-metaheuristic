""" A collection of binary-valued functions for benchmark optimisation algorithms """
import numpy as np

## Benchmark definitions (numpy vectorised implementation) ##

## Real-valued problems ##
def sphere(X):
    return np.sum(X**2, axis=1)

def rosenbrock(X):
    return (1 - X[:,0])**2 + 100 * (X[:,1] - (X[:,0]**2))**2

def rastrigin(X):
    d = X.shape[1]      # Search space dimensionality
    return np.sum((X**2 - 10*np.cos(2*np.pi*X)), axis=1) + 10*d

def rastrigin_bipolar(X):
    d = X.shape[1]      # Search space dimensionality
    return np.sum(((X-(2*(np.array(range(0,d))%2)-1))**2 - 10*np.cos(2*np.pi*(X-(2*(np.array(range(0,d))%2)-1)))), axis=1) + 10*d

def rastrigin_offset(X):
    d = X.shape[1]      # Search space dimensionality
    return np.sum(((X-1.123)**2 - 10*np.cos(2*np.pi*(X-1.123))), axis=1) + 10*d

def himmelblau(X):
    return (X[:,0]**2 + X[:,1] - 11)**2 + (X[:,0] + X[:,1]**2 - 7)**2

def eggholder(X):
    Z = X[:,1]+47
    return (-Z * np.sin(np.sqrt((np.abs(X[:,0]/2 + Z)))) \
            -X[:,0] * np.sin(np.sqrt((np.abs(X[:,0] - Z))))) #+ 959.640662720851


## Binary problems ##

def oneMax(B):
    """ oneMax counts the numbers of bits set in a given bitstring. Its maximum occurs when all d bits are set to 1.
        NB. Since optimiser minimises by default, the sum is negated for maximisation purposes.
        Arguments:
        B -- a population of bitstrings of length d
    """
    return -np.sum(B, axis=1)

def squareWave(B):
    """ squareWave computes the similarity of a bitstring to a periodic waveform of length d and period tau
        consisting of instantaneous transitions between levels 0 and 1 every tau/2 bits. It is basically a discrete
        version of a sin waveform, indeed it can be think of as the sign of the sin.
        NB. Since optimiser minimises by default, the similarity is negated for maximisation purposes.
        Arguments:
        B -- a population of bitstrings of length d. It is suggested d being a perfect square so that tau = sqrt(d)
    """
    d = B.shape[1]; tau = int(np.sqrt(d))
    S = -1 * (2 * (np.arange(d) // tau) - (2 * np.arange(d) // tau))
    return np.array([-np.sum(B_ == S) for B_ in B])

def binVal(B):
    """ binVal obtains the decimal value of a given bitstring. Its maximum occurs when all d bits are set to 1.
        NB. Since optimiser minimises by default, the value is negated for maximisation purposes.
        Arguments:
        B -- a population of bitstrings of length d
    """
    d = B.shape[1]
    # In the following line, use dtype=float to avoid overflow when d>=64 bits
    return -np.sum(np.multiply(2**np.arange(d, dtype=np.float)[::-1], B), dtype=np.float, axis=1)

def powSum(B):
    """ powSum obtains the sum of the exponents of the powers of two (or loci) that are set to one in a bitstring.
        Its maximum occurs when all d bits are set to 1.
        NB. Since optimiser minimises by default, the value is negated for maximisation purposes.
        Arguments:
        B -- a population of bitstrings of length d
    """
    d = B.shape[1]
    # In the following line, use dtype=float to avoid overflow when d>=64 bits
    return -np.sum(np.multiply(np.arange(1, d+1, dtype=np.float)[::-1], B), dtype=np.float, axis=1)

def knapsack_neglect(B):
    """ knapsack_neglect """
    if (weight_item * B) <= capacity_knapsack:

        return np.sum((profit_item*B), axis=1)
    else:
        return -10000000
def knapsack_penalty(B):
    """ knapsack_penalty """
    return (np.sum((profit_item*B), axis=1) - (np.sum((weight_item), axis=1)*abs(-capacity_knapsack+np.sum((weight_item*B), axis=1))))
