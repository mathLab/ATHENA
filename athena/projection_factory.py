"""
Module for the factory class for projection matrices
"""
import numpy as np


class classproperty():
    """
    Custom decorator.
    """
    def __init__(self, f):
        self.f = f
        self.__doc__ = f.__doc__
        self.__name__ = f.__name__

    def __get__(self, obj, owner):
        return self.f(owner)


class ProjectionFactory():
    """
    Factory class that spawns projection matrices.

    :Example:

        >>> from athena import ProjectionFactory
        >>> import numpy as np
        >>> input_dim = 2
        >>> n_features = 40
        >>> params = [1.1, 2.9]
        >>> for pname in ProjectionFactory.projections:
        >>>     y = ProjectionFactory(pname)(input_dim, n_features, params)
    """
    @staticmethod
    def beta(input_dim, n_features, params):
        """
        Beta distribution

        :param int input_dim: dimension of the inputs.
        :param int n_features: dimension of the RKHS.
        :param list params: the two parameters are the alpha and beta
            shape parameters respectively.
        :return: n_features-by-input_dim projection matrix.
        :rtype: numpy.ndarray.
        """
        return np.random.beta(params[0], params[1], (n_features, input_dim))

    @staticmethod
    def cauchy(input_dim, n_features, params):
        """
        Cauchy distribution

        :param int input_dim: dimension of the inputs.
        :param int n_features: dimension of the RKHS.
        :param list params: the single parameter is a scale factor.
        :return: n_features-by-input_dim projection matrix.
        :rtype: numpy.ndarray.
        """
        return (1 / params[0]) * np.random.standard_cauchy(
            (n_features, input_dim))

    @staticmethod
    def dirichlet(input_dim, n_features, params):
        """
        Dirichlet distribution

        :param int input_dim: dimension of the inputs.
        :param int n_features: dimension of the RKHS.
        :param list params: the single parameter is a scale to the input_dim
            dimensional shape parameter.
        :return: n_features-by-input_dim projection matrix.
        :rtype: numpy.ndarray.
        """
        return np.random.dirichlet(params[0] * np.ones(input_dim), n_features)

    @staticmethod
    def laplace(input_dim, n_features, params):
        """
        Laplace distribution

        :param int input_dim: dimension of the inputs.
        :param int n_features: dimension of the RKHS.
        :param list params: the single parameter is the scale of the
            distribution, the mean is set to 0.
        :return: n_features-by-input_dim projection matrix.
        :rtype: numpy.ndarray.
        """
        return np.random.laplace(0, params[0], (n_features, input_dim))

    @staticmethod
    def multivariate_normal(input_dim, n_features, params):
        """
        Multivariate normal distribution

        :param int input_dim: dimension of the inputs.
        :param int n_features: dimension of the RKHS.
        :param list params: the input_dim dimensioanl parameters are the
        diagonal of the covariance matrix of the distribution. The mean is set
        to the 0 vector.
        :return: n_features-by-input_dim projection matrix.
        :rtype: numpy.ndarray.
        """
        return np.random.multivariate_normal(np.zeros(input_dim),
                                             np.diag(params), n_features)

    @staticmethod
    def normal(input_dim, n_features, params):
        """
        Normal distribution

        :param int input_dim: dimension of the inputs.
        :param int n_features: dimension of the RKHS.
        :param list params: the single parameter is the variance of the
            distribution. The mean is set to 0.
        :return: n_features-by-input_dim projection matrix.
        :rtype: numpy.ndarray.
        """
        return np.random.normal(0, params[0], (n_features, input_dim))

    @staticmethod
    def uniform(input_dim, n_features, params):
        """
        Uniform distribution

        :param int input_dim: dimension of the inputs.
        :param int n_features: dimension of the RKHS.
        :param list params: the two parameters are the extremals of the domain.
        :return: n_features-by-input_dim projection matrix.
        :rtype: numpy.ndarray.
        """
        return np.random.uniform(params[0], params[1], (n_features, input_dim))

    ###########################################################################
    ##                                                                       ##
    ## PROJECTION FUNCTION dictionary                                        ##
    ##                                                                       ##
    ## New implementations must be added here.                               ##
    ##                                                                       ##
    ###########################################################################
    __projections = {
        'beta': beta.__func__,
        'cauchy': cauchy.__func__,
        'dirichlet': dirichlet.__func__,
        'laplace': laplace.__func__,
        'multivariate_normal': multivariate_normal.__func__,
        'normal': normal.__func__,
        'uniform': uniform.__func__
    }

    def __new__(cls, fname):
        # to make the str callable we have to use a dictionary with all the
        # implemented projection matrices
        if fname in cls.projections:
            return cls.__projections[fname]
        raise NameError(
            """The name of the projection matrix is not correct or not
            implemented. Check the documentation for all the available
            possibilities.""")

    @classproperty
    def projections(self):
        """
        The available projection matrices.

        :return: the list of all the available projection matrices.
        :rtype: list
        """
        return list(self.__projections.keys())
