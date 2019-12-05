"""[summary]

[description]
"""
import numpy as np


def initialize_weights(matrix):
    return np.ones((matrix.shape[0], 1)) / matrix.shape[0]


def local_linear_gradients(inputs, outputs, weights=None, n_neighbors=None):
    """Estimate a collection of gradients from input/output pairs.
    
    Given a set of input/output pairs, choose subsets of neighboring points and
    build a local linear model for each subset. The gradients of these local
    linear models comprise estimates of sampled gradients.
    Parameters
    ----------
    inputs : ndarray 
        M-by-m matrix that contains the m-dimensional inputs
    outputs : ndarray 
        M-by-1 matrix that contains scalar outputs
    n_neighbors : int, optional
        how many nearest neighbors to use when constructing the local linear 
        model (default 1)
    weights : ndarray, optional
        M-by-1 matrix that contains the weights for each observation (default 
        None)
    Returns
    -------
    gradients : ndarray
        M-by-m matrix that contains estimated partial derivatives approximated 
        by the local linear models
    Notes
    -----
    If `n_neighbors` is not specified, the default value is floor(1.7*m).
    """
    n_samples, n_pars = inputs.shape

    if n_samples <= n_pars:
        raise Exception('Not enough samples for local linear models.')
    if n_neighbors is None:
        n_neighbors = int(min(np.floor(1.7 * n_pars), n_samples))
    elif not isinstance(n_neighbors, int):
        raise TypeError('n_neighbors ({}) must be an integer.'.format(
            n_neighbors))

    if n_neighbors < n_pars + 1 or n_neighbors > n_samples:
        raise Exception(
            'n_neighbors ({}) must be between the number of parameters ({}) and the number of samples ({})'.
            format(n_neighbors, n_pars, n_samples))

    if weights is None:
        weights = initialize_weights(inputs)

    MM = min(int(np.ceil(10 * n_pars * np.log(n_pars))), n_samples - 1)
    gradients = np.zeros((MM, n_pars))
    for i in range(MM):
        ii = np.random.randint(n_samples)
        inputs_rand_row = inputs[ii, :]
        D2 = np.sum((inputs - inputs_rand_row)**2, axis=1)
        ind = np.argsort(D2)
        ind = ind[D2 != 0]
        A = np.hstack((np.ones((n_neighbors, 1)),
                       inputs[ind[:n_neighbors], :])) * np.sqrt(weights[ii])
        b = outputs[ind[:n_neighbors]] * np.sqrt(weights[ii])
        u = np.linalg.lstsq(A, b, rcond=None)[0]
        gradients[i, :] = u[1:].T

    return gradients


def sort_eigpairs(matrix):
    """Compute eigenpairs and sort.
    
    Parameters
    ----------
    C : ndarray
        matrix whose eigenpairs you want
        
    Returns
    -------
    e : ndarray
        vector of sorted eigenvalues
    W : ndarray
        orthogonal matrix of corresponding eigenvectors
    
    Notes
    -----
    Eigenvectors are unique up to a sign. We make the choice to normalize the
    eigenvectors so that the first component of each eigenvector is positive.
    This normalization is very helpful for the bootstrapping. 
    """
    evals, evects = np.linalg.eigh(matrix)
    evals = abs(evals)
    ind = np.argsort(evals)
    evals = evals[ind[::-1]]
    evects = evects[:, ind[::-1]]
    s = np.sign(evects[0, :])
    s[s == 0] = 1
    evects *= s
    return evals.reshape((evals.size, 1)), evects
