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
<<<<<<< HEAD
    return -np.sum(np.multiply(np.arange(1, d+1, dtype=np.float)[::-1], B), dtype=np.float, axis=1)
=======
    return -np.sum(np.multiply(np.arange(1, d + 1, dtype=np.float)[::-1], B), dtype=np.float, axis=1)


############################
## Combinatorial problems ##
############################

# Knapsack Problem #

def knapsack_instance(zf,filename):
    """ knapsack_instance loads from a file the parameters of an instance of KP problem
        The data is stored in a global dictionary.
        Arguments:
        filname -- the name of the file with the instance data
        Output:
        kp_data -- a global dictionary containing the parameters of a KP instance
    """
    #leer del archivo en el .zip los parámetros de prueba#
    if "l-d" in filename:
        parameters = np.loadtxt(zf.open(filename), dtype=float)
        n_items = int(np.array(parameters[0,0]))
        C_max = np.full((1, 1), float(parameters[0, 1]))[0]
        profits = np.array(parameters[1:, 0])
        weights = np.array(parameters[1:, 1])
        v_optimum = 0
        time_optimum = 0
    else:
        general_parameters = pd.read_csv(zf.open(filename), delimiter=" ", nrows=4)
        n_items = int(general_parameters.iloc[0])
        C_max = np.array(general_parameters.iloc[1],dtype=float)
        v_optimum = np.array(general_parameters.iloc[2])
        time_optimum = np.array(general_parameters.iloc[3])
        parameters_item_ = pd.read_csv(zf.open(filename), delimiter=",", nrows=n_items, skiprows=5,
                            names=["n","profits", "weights","B_optimum"])
        weights = np.array(parameters_item_.weights,dtype=float)
        profits = np.array(parameters_item_.profits,dtype=float)
    global kp_data
    kp_data = {"d": n_items, "C_max": C_max, "weights": weights, "profits": profits,"v_optimum": v_optimum, "time_optimum":time_optimum}
    return n_items, v_optimum, time_optimum

def knapsack_discard(B):
    """ knapsack_discard obtains fitnesses using the KP cost function, except on those candidates violating
        the capacity constraint. The unfeasible candidates are "discarded" by setting their fitness to -Inf.
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
    discard_mask = (C > C_max)  # Filter out those violating the C_max constraint
    P[discard_mask] = -np.Inf  # Discard them
    return P.max()  # Return best profit

def knapsack_penalty(B):
    """ knapsack_discard obtains fitnesses using the KP cost function, except on those candidates violating
        the capacity constraint. The unfeasible candidates are "penalty" by setting their fitness to -Inf.
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
    P=np.sum(np.multiply(profits, B), axis=1)
    discard_mask = (C > C_max)  # Filter out those violating the C_max constraint
    P[discard_mask] = (C_max-C[discard_mask]) # Penalty them
    return P.max()  # Return best profit
<<<<<<< HEAD
>>>>>>> parent of 2202868... penalty
=======
>>>>>>> parent of 2202868... penalty
