## Needed libraries ##
import numpy as np

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
