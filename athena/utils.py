"""Utility functions module.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from sklearn.gaussian_process import GaussianProcessRegressor


class Normalizer():
    """A class for normalizing and unnormalizing bounded inputs.

    :param numpy.ndarray lb: array n_params-by-1 that contains lower bounds
        on the simulation inputs.
    :param numpy.ndarray ub: array n_params-by-1 that contains upper bounds
        on the simulation inputs.
    """
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def fit_transform(self, inputs):
        """Return corresponding points shifted and scaled to [-1, 1]^n_params.

        :param numpy.ndarray inputs: contains all input points to normalize.
            The shape is n_samples-by-n_params. The components of each row of
            `inputs` should be between `self.lb` and `self.ub`.
        :return: the normalized inputs. The components of each row should be
            between -1 and 1.
        :rtype: numpy.ndarray
        """
        return 2.0 * (inputs - self.lb) / (self.ub - self.lb) - 1.0

    def inverse_transform(self, inputs):
        """Return corresponding points shifted and scaled to
        `[self.lb, self.ub]`.

        :param numpy.ndarray inputs: contains all input points to unnormalize.
            The shape is n_samples-by-n_params. The components of each row of
            `inputs` should be between -1 and 1.
        :return: the unnormalized inputs. The components of each row should be
            between `self.lb` and `self.ub`.
        :rtype: numpy.ndarray
        """
        return (self.ub - self.lb) * (inputs + 1.0) / 2.0 + self.lb


def initialize_weights(matrix):
    """
    Inizialize uniform weights for simple Monte Carlo method or linear regression in
    local linear gradients.

    :param numpy.ndarray matrix: matrix which shape[0] value contains the
        dimension of the weights to be computed.
    :return: weights
    :rtype: numpy.ndarray
    """
    return np.ones((matrix.shape[0], 1)) / matrix.shape[0]


def linear_program_ineq(c, A, b):
    """Solves an equality constrained linear program with variable bounds.
    This method returns the minimizer of the following linear program.

    minimize  c^T x
    subject to  A x >= b

    :param numpy.ndarray c: coefficients vector of the linear objective
        function to be minimized.
    :param numpy.ndarray A: 2-D array which, when matrix-multiplied by x,
        gives the values of the lower-bound inequality constraints at x.
    :param numpy.ndarray b: 1-D array of values representing the lower-bound
        of each inequality constraint (row) in A.
    :return: the independent variable vector which minimizes the linear
        programming problem.
    :rtype: numpy.ndarray
    :raises: RuntimeError
    """
    c = c.reshape(-1, )
    b = b.reshape(-1, )

    # make unbounded bounds
    bounds = [(None, None) for _ in range(c.shape[0])]
    res = linprog(c=c, A_ub=-A, b_ub=-b, bounds=bounds)
    if res.success:
        return res.x.reshape(-1, 1)
    raise RuntimeError(f'Scipy did not solve the LP. {res.message}')


def local_linear_gradients(inputs, outputs, weights=None, n_neighbors=None):
    """Estimate a collection of gradients from input/output pairs.

    Given a set of input/output pairs, choose subsets of neighboring points and
    build a local linear model for each subset. The gradients of these local
    linear models comprise estimates of sampled gradients.

    :param numpy.ndarray inputs:
        M-by-m matrix that contains the m-dimensional inputs
    :param numpy.ndarray outputs:
        M-by-1 matrix that contains scalar outputs
    :param numpy.ndarray weights:
        M-by-1 matrix that contains the weights for each observation (default
        None)
    :param int n_neighbors:
        how many nearest neighbors to use when constructing the local linear
        model. the default value is floor(1.7*m)
    :return: M-by-m matrix that contains estimated partial derivatives approximated
        by the local linear models; the corresponding new inputs
    :rtype: numpy.ndarray, numpy.ndarray

    :raises: ValueError, TypeError
    """
    n_samples, n_pars = inputs.shape

    if n_samples <= n_pars:
        raise ValueError('Not enough samples for local linear models.')

    if n_neighbors is None:
        n_neighbors = int(min(np.floor(1.7 * n_pars), n_samples))
    elif not isinstance(n_neighbors, int):
        raise TypeError(f'n_neighbors ({n_neighbors}) must be an integer.')

    if n_neighbors <= n_pars or n_neighbors > n_samples:
        raise ValueError(
            f'n_neighbors must be between the number of parameters and the number of samples. Unsatisfied: {n_pars} < {n_neighbors} < {n_samples}.'
        )

    if weights is None:
        weights = initialize_weights(inputs)

    MM = min(int(np.ceil(10 * n_pars * np.log(n_pars))), n_samples - 1)

    # distinguish between scalar and vectorial outputs
    if len(outputs.shape) == 1 or outputs.shape[1] == 1:
        gradients = np.zeros((MM, n_pars))
    else:
        gradients = np.zeros((MM, outputs.shape[1], n_pars))

    # new inputs are defined since MM is different from n_samples
    new_inputs = np.zeros((MM, n_pars))

    for i in range(MM):
        ii = np.random.randint(n_samples)
        inputs_rand_row = inputs[ii, :]
        D2 = np.sum((inputs - inputs_rand_row)**2, axis=1)
        ind = np.argsort(D2)
        ind = ind[D2 != 0]
        A = np.hstack((np.ones(
            (n_neighbors, 1)), inputs[ind[:n_neighbors], :])) * np.sqrt(
                weights[ii])
        b = outputs[ind[:n_neighbors]] * np.sqrt(weights[ii])
        u = np.linalg.lstsq(A, b, rcond=None)[0]
        gradients[i] = u[1:].T
        new_inputs[i, :] = (1 / n_neighbors) * np.sum(
            inputs[ind[:n_neighbors], :], axis=0)
    return gradients, new_inputs


def sort_eigpairs(evals, evects):
    """Sort eigenpairs.

    :param numpy.ndarray evals: eigenvalues.
    :param numpy.ndarray evects: eigenvectors.
    :return: vector of sorted eigenvalues; orthogonal matrix of corresponding
        eigenvectors.
    :rtype: numpy.ndarray, numpy.ndarray

    .. note::

        Eigenvectors are unique up to a sign. We make the choice to normalize
        the eigenvectors so that the first component of each eigenvector is
        positive. This normalization is very helpful for the bootstrapping.
    """
    evals = abs(evals)
    ind = np.argsort(evals)
    evals = evals[ind[::-1]]
    evects = evects[:, ind[::-1]]
    s = np.sign(evects[0, :])
    s[s == 0] = 1
    evects *= s
    return evals, evects


class CrossValidation():
    """
    Class to perform k-fold cross validation when tuning hyperparameters for the
    design of a response surface with ActiveSubspaces or KernelActiveSubspaces.
    Used in particular during the tuning of the parameters of the spectral
    distribution of the feature map, inside the object function average_rrmse.
    default score is the relative root mean square error (rrmse).

    :param numpy.ndarray inputs: n_samples-by-input_dim input matrix.
    :param numpy.ndarray outputs: n_sample-by-output_dim output matrix.
    :param numpy.ndarray gradients: n_samples-by-output_dim-by-input_dim
        gradients matrix.
    :param `Subspaces` subspace: ActiveSubspace or KernelActiveSubspace object,
        from which evaluate the response surface. The dimension of the response
        surface is specified in subspace.dim attribute.
    :param int folds: number of folds of the cross-validation procedure.
    :param dict kwargs: additional paramters organized in a dictionary to pass
        to subspace.fit method. For example 'weights' or 'metric'.

    :cvar `sklearn.gaussian_process.GaussianProcessRegressor` gp: Gaussian
        process of the response surface built with scikit-learn.
    """
    def __init__(self, inputs, outputs, gradients, subspace, folds=5, **kwargs):

        if any(v is None for v in [inputs, outputs, gradients, subspace]):
            raise ValueError(
                'Any among inputs, outputs, gradients, subspace is None.')

        self.inputs = inputs
        self.outputs = outputs
        self.gradients = gradients
        self.ss = subspace
        self.folds = folds
        self.gp = None
        self.kwargs = kwargs

    def run(self):
        """
        Run the k-fold cross validation procedure. In each fold a fit and an
        evaluation of the score are compute.

        :return: mean and standard deviation of the scores.
        :rtype: list of two numpy.ndarray.
        """
        mask = np.arange(self.inputs.shape[0])

        np.random.shuffle(mask)
        scores = np.zeros((self.folds))
        s_mask = np.array_split(mask, self.folds)

        for i in range(self.folds):
            v_mask = s_mask[i]
            validation = (self.inputs[v_mask, :], self.outputs[v_mask, :])

            t_mask = ~v_mask
            self.fit(self.inputs[t_mask, :], self.gradients[t_mask, :, :],
                     self.outputs[t_mask, :])

            scores[i] = self.scorer(validation[0], validation[1])

        return scores.mean(), scores.std()

    def fit(self, inputs, gradients, outputs):
        """
        Uses Gaussian process regression to build the response surface as a side
        effect. The dimension of the response surface is specified in the
        attribute self.ss.dim.

        :param numpy.ndarray inputs: n_samples-by-input_dim input matrix.
        :param numpy.ndarray outputs: n_sample-by-output_dim output matrix.
        :param numpy.ndarray gradients: n_samples-by-output_dim-by-input_dim
            gradients matrix.
        """
        self.ss.fit(inputs=inputs,
                    gradients=gradients,
                    outputs=outputs,
                    **self.kwargs)
        y = self.ss.transform(inputs)[0]

        self.gp = GaussianProcessRegressor(n_restarts_optimizer=15)
        self.gp.fit(y, np.atleast_2d(outputs))

    def predict(self, inputs):
        """
        Predict method of cross-validation.

        :param numpy.ndarray inputs: n_samples-by-input_dim input matrix.
        :return: n_samples-by-dim prediction of the surrogate response surface
            model at the inputs. The value dim corresponds to self.ss.dim.
        :rtype: numpy.ndarray
        """
        x_test = self.ss.transform(inputs)[0]
        return self.gp.predict(np.atleast_2d(x_test), return_std=False)

    def scorer(self, inputs, outputs):
        """
        Score function of cross-validation.

        :param numpy.ndarray inputs: n_samples-by-input_dim input matrix.
        :param numpy.ndarray outputs: n_sample-by-output_dim output matrix.
        :return: relative root mean square error between inputs and outputs.
        :rtype: np.float64
        """
        y = self.predict(inputs)
        return rrmse(y, outputs)


def rrmse(predictions, targets):
    """
    Evaluates the relative root mean square error. It can be vectorized for
    multidimensional predictions and targets.

    :param numpy.ndarray predictions: predictions input.
    :param numpy.ndarray targets: targets input.
    :return: relative root mean squared error
    :rtype: np.float64
    """
    n_samples = predictions.shape[0]
    if n_samples != targets.shape[0]:
        raise ValueError('Predictions and targets differ in number of samples.')

    t = np.atleast_2d(targets).reshape(n_samples, -1)
    p = np.atleast_2d(predictions).reshape(n_samples, -1)
    std_deviation = np.linalg.norm(t - np.mean(t, axis=0).reshape(1, -1))
    return np.linalg.norm(p - t) / std_deviation


def average_rrmse(hyperparams, best, csv, verbose=False, resample=5):
    """
    Objective function to be optimized during the tuning process of the method
    :func:`~athena.FeatureMap.tune_pr_matrix`. The optimal hyperparameters of the
    spectral distribution are searched for in a domain logarithmically scaled in
    base 10. For each call of :func:`~athena.utils.average_rrmse` by the
    optimizer, the same hyperparameter is tested in two nested procedures: in
    the external procedure the projection matrix is resampled a number of times
    specified by the resample parameter; in the internal procedure the relative
    root mean squared error (:func:`~athena.utils.rrmse`) is evaluated as the
    k-fold mean of a k-fold cross-validation procedure. The score of a single
    fold of this cross-validation procedure is the rrmse on the validation set
    of the predictions of the response surface built with a Subspace object on
    the training set.

    :param list hyperparameters: logarithm of the parameter of the spectral
        distribution passed to average_rrmse by the optimizer.
    :param 'CrossValidation' csv: CrossValidation object which contains the
        same Subspace object and the inputs, outputs, gradients datasets. The
    :param list best: list that records the best score and the best
        projection matrix. The initial values are 0.8 and a
        n_features-by-input_dim numpy.ndarray of zeros.
    :param int resample: number of times the projection matrix is resampled
        from the same spectral distribution with the same hyperparameter.
    :param bool verbose: True to print the score for each resample.
    :return: minumum of the scores evaluated for the same hyperparameter and
        a specified number of resamples of the projection matrix.
    :rtype: numpy.float64
    """
    if not isinstance(csv, CrossValidation):
        raise ValueError(
            "The argument csv must be of type athena.utils.CrossValidation")

    if len(hyperparams.shape) > 1:
        hyperparams = np.squeeze(hyperparams)
    if len(hyperparams.shape) == 0:
        hyperparams = np.array([hyperparams])

    # compute the real hyperparameters
    hyperparams = 10**hyperparams

    # list of scores for the same hyperparameters but different samples
    # of the projection matrix
    score_records = []

    #set the hyperparameters
    csv.ss.feature_map.params = hyperparams

    if verbose is True:
        print("#" * 80)
    for _ in range(resample):
        # compute the score with cross validation for the sampled projection
        # matrix
        mean, std = csv.run()

        # save the best parameters
        if verbose is True:
            print(f"params {hyperparams} mean {mean}, std {std}")
        score_records.append(mean)

        # skip resampling from the same hyperparam if the error is not below
        # the treshold 0.8
        if mean > 0.8:
            break
        if mean <= best[0]:
            best[0] = mean
            best[1] = csv.ss.feature_map.pr_matrix

        # set _pr_matrix to None so that csv.ss.feature_map.compute_fmap
        # and csv.ss.feature_map.compute_fmap_jac resample the projection matrix
        # for the same hyperparams
        csv.ss.feature_map._pr_matrix = None
    return min(score_records)
