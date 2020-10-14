'''
A collection of benchmark optimisation problems to be tested with PUPI
PUPI algorithm is a method for unconstrained cost function optimisation
inspired in the foraging behaviour of urban park pigeons.
(c) 2020 - v3.0 (October/2020)
License: GPLv3 (see https://www.gnu.org/licenses/gpl-3.0.txt)
'''

import numpy as np

## Benchmark definitions (numpy vectorised implementation) ##

##########################
## Real-valued problems ##
##########################

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

############################################
## Additional real-valued problems (v3.0) ##
############################################

def ackley(x):
    d = x.shape[1]     # Search space dimensionality
    return (-20.0 * np.exp(-0.2 * np.sqrt((1 / d) * (x ** 2).sum(axis=1)))
            - np.exp((1 / float(d)) * np.cos(2 * np.pi * x).sum(axis=1))
            + 20.0
            + np.exp(1))

def beale(x):
    return ((1.5 - x[:, 0] + x[:, 0] * x[:, 1]) ** 2.0
            + (2.25 - x[:, 0] + x[:, 0] * x[:, 1] ** 2.0) ** 2.0
            + (2.625 - x[:, 0] + x[:, 0] * x[:, 1] ** 3.0) ** 2.0)

def booth(x):
    return (x[:, 0] + 2 * x[:, 1] - 7) ** 2.0 + (2 * x[:, 0] + x[:, 1] - 5) ** 2.0

def crossintray(x):
    return -0.0001 * np.power(
        np.abs(
            np.sin(x[:, 0])
            * np.sin(x[:, 1])
            * np.exp(np.abs(100 - (np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) / np.pi)))
        )
        + 1,
        0.1,)

def easom(x):
    return (-1
            * np.cos(x[:, 0])
            * np.cos(x[:, 1])
            * np.exp(-1 * ((x[:, 0] - np.pi) ** 2 + (x[:, 1] - np.pi) ** 2)))

def goldstein(x):
    return (1 + (x[:, 0] + x[:, 1] + 1) ** 2.0 * (19 - 14 * x[:, 0] + 3 * x[:, 0] ** 2.0
                                                  - 14 * x[:, 1] + 6 * x[:, 0] * x[:, 1] + 3 * x[:, 1] ** 2.0)) \
        * (30 + (2 * x[:, 0] - 3 * x[:, 1]) ** 2.0 * (18 - 32 * x[:, 0] + 12 * x[:, 0] ** 2.0 + 48 * x[:, 1]
                                                      - 36 * x[:, 0] * x[:, 1] + 27 * x[:, 1] ** 2.0))

def holdertable(x):
    return -np.abs(
        np.sin(x[:, 0])
        * np.cos(x[:, 1])
        * np.exp(np.abs(1 - np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) / np.pi)))

def levy(x):
    mask = np.full(x.shape, False)
    mask[:, -1] = True
    masked_x = np.ma.array(x, mask=mask)
    w_ = 1 + (x - 1) / 4
    masked_w_ = np.ma.array(w_, mask=mask)
    d_ = x.shape[1] - 1
    return (np.sin(np.pi * w_[:, 0]) ** 2.0
            + ((masked_x - 1) ** 2.0).sum(axis=1)
            * (1 + 10 * np.sin(np.pi * masked_w_.sum(axis=1) + 1) ** 2.0)
            + (w_[:, d_] - 1) ** 2.0 * (1 + np.sin(2 * np.pi * w_[:, d_]) ** 2.0))

def matyas(x):
    return 0.26 * (x[:, 0] ** 2.0 + x[:, 1] ** 2.0) - 0.48 * x[:, 0] * x[:, 1]

def schaffer2(x):
    return 0.5 + ((np.sin(x[:, 0] ** 2.0 - x[:, 1] ** 2.0) ** 2.0 - 0.5)
               / ((1 + 0.001 * (x[:, 0] ** 2.0 + x[:, 1] ** 2.0)) ** 2.0))

def threehump(x):
    return 2 * x[:, 0] ** 2 - 1.05 * (x[:, 0] ** 4) + (x[:, 0] ** 6) / 6 + x[:, 0] * x[:, 1] + x[:, 1] ** 2

def bohachevsky1(x):
    return x[:, 0] ** 2.0 + 2.0 * x[:, 1] ** 2.0 - 0.3 * np.cos(3 * np.pi * x[:, 0]) \
           - 0.4 * np.cos(4 * np.pi * x[:, 1]) + 0.7

def zakharov(x):
    d = x.shape[1]		# Search space dimensionality
    return np.sum(x ** 2, axis=1) + (0.5 * np.sum(np.arange(1, d+1) * x, axis=1)) ** 2 \
           + (0.5 * np.sum(np.arange(1, d+1) * x, axis=1)) ** 4

def dixonprice(x):
    d = x.shape[1]		# Search space dimensionality
    return (x[:, 0] - 1) ** 2 + np.sum(np.arange(2, d+1) * (2.0 * x[:, 1:] ** 2.0 - x[:, -1]) ** 2.0, axis=1)

def michalewicz(x):
    d = x.shape[1]		# Search space dimensionality
    return -np.sum(np.sin(x) * (np.sin( np.arange(1, d) * x ** 2 / np.pi)) ** 20, axis=1)

def mishra3(x):
    return np.sqrt(np.abs(np.cos(np.sqrt(np.abs(x[:, 0] ** 2 + x[:, 1] ** 2))))) + 0.01 * (x[:, 0] + x[:, 1])

def mishra5(x):
    return (np.sin((np.cos(x[:, 0]) + np.cos(x[:, 1])) ** 2) ** 2
            + np.cos((np.sin(x[:, 0]) + np.sin(x[:, 1])) ** 2) ** 2 + x[:, 0]) ** 2 + 0.01 * (x[:, 0] + x[:, 1])

def mishra6(x):
    a = (np.cos(x[:, 0]) + np.cos(x[:, 1])) ** 2
    b = (np.sin(x[:, 0]) + np.sin(x[:, 1])) ** 2
    c = 0.1 * ((x[:, 0] - 1) ** 2 + (x[:, 1] - 1) ** 2)
    return c - np.log((np.sin(a) ** 2 - np.cos(b) ** 2 + x[:, 0]) ** 2)


def dropwave(x):
    return - (1 + np.cos(12 * np.sqrt(np.sum(x**2, axis=1)))) / (0.5 * (np.sum(x**2, axis=1)) + 2)

def hosaki(x):
    return (1 - 8 * x[:, 0] + 7 * x[:, 0] ** 2 - (7 / 3) * x[:, 0] ** 3 + - (1 / 4) * x[:, 0] ** 4) \
           * (x[:, 1] ** 2) * np.exp(-x[:, 0])

def damavandi(x):
    return (1 - np.abs((np.sin(np.pi * (x[:, 0] - 2)) * np.sin(np.pi * (x[:, 1] - 2))) /
                       (np.pi ** 2 * ((x[:, 0] - 2) * (x[:, 1] - 2)))) ** 5) \
           * (2 + (x[:, 0] - 7) ** 2 + 2 * (x[:, 1] - 7) ** 2)

def parsopoulos(x):
    return  np.cos(x[:, 0]) ** 2 + np.sin(x[:, 1]) ** 2

def vincent(x):
    return -np.sum(np.sin(10 * np.log(x)), axis=1)

#####################
## Binary problems ##
#####################

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


###################################
## Combinatorial problems (v3.0) ##
###################################

# Knapsack Problem #

def knapsack_instance(filename):
    """ knapsack_instance loads from a file the parameters of an instance of KP problem
        The data is stored in a global dictionary.
        Arguments:
        filname -- the name of the file with the instance data

        Output:
        kp_data -- a global dictionary containing the parameters of a KP instance
    """
    file_ = open(filename)
    line_ = file_.readline().split()
    n_items, C_max = int(line_[0]), float(line_[1])
    data_ = np.loadtxt(filename, skiprows=1, dtype=float)
    profits = np.array(data_[:, 0])
    weights = np.array(data_[:, 1])

    global kp_data
    kp_data = {"n_items": n_items, "C_max": C_max, "weights": weights, "profits": profits}
    print("KP instance loaded!", kp_data)
    return kp_data


def knapsack_discard(B):
    """ knapsack_discard obtains fitnesses using the KP cost function, except on those candidates violating
        the capacity constraint. The unfeasible candidates are "discarded" by setting their fitness to -Inf.
        NB. Since optimiser minimises by default, the value is negated for maximisation purposes.

        Arguments:
        B -- a population of bitstrings of length d
        kp_data -- a global dictionary containing the parameters of a KP instance
    """
    # First we need to retrieve the KP instance #
    global kp_data
    C_max = kp_data["C_max"]
    weights = kp_data["weights"]
    profits = kp_data["profits"]

    # Now compute the costs obtained with each candidate
    C = np.sum(np.multiply(weights, B), axis=1)  # Capacity of all candidates
    P = np.sum(np.multiply(profits, B), axis=1)  # Profitability of all candidates
    discard_mask = (C > C_max)                   # Filter out those violating the C_max constraint
    P[discard_mask] = -np.Inf                  # Discard them
    return -P                                    # Return profits

