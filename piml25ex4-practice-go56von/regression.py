###########################################################################
#               Physics-Informed Machine Learning                         #
#                             SS 2023                                     #
#                                                                         #
#                     Exercise 5 - Template                               #
#                                                                         #
# NOTICE: Sharing and distribution of any course content, other than      #
# between individual students registered in the course, is not permitted  #
# without permission.                                                     #
#                                                                         #
###########################################################################

import numpy as np


def basis_functions_matrix(X):
    """Compute the basis matrix, which contains the basis functions in the columns

    Parameters
    ----------
    X : array, shape [N, D]
            Input matrix

    Returns
    -------
    Phi : array [N, ?]

    """
    symbolic_representation_list = None
    Phi = None
    ##############
    ##############


    # TODO

    symbolic_representation_list = ["1", "r", "f", "r^2", "r*f", "f^2", "r^3", "f^3", "sin(r)", "sin(f)", "cos(r)", "cos(f)"]
    poly_2_matrix = np.concatenate(
        [X[:, 0].reshape(-1, 1) ** 2, X[:, 0].reshape(-1, 1) * X[:, 1].reshape(-1, 1), X[:, 1].reshape(-1, 1) ** 2],
        axis=1) #colums, respectively: r**2,r*f,f**2
    Phi = np.concatenate([np.ones([X.shape[0], 1]), X, poly_2_matrix, X ** 3, np.sin(X), np.cos(X)], axis=1) #


    ##############
    ##############
    return Phi, symbolic_representation_list


def MSE(y_true, y_pred):
    """Compute mean squared error between true and predicted regression outputs.

    Parameters
    ----------
    y_true : np.array
        True regression outputs.
    y_pred : np.array
        Predicted regression outputs.

    Returns
    -------
    mse : float
        Mean squared error.

    """

    return np.mean((y_true - y_pred) ** 2)


class LeastSquares(object):
    def __init__(self):
        self.weights = None
        self.rho = 0
        self.z = 0

    def fit_lasso(self, Phi, y, lambda_val, iterations=100, weights_init=None, tol=1e-3):
        """Fit the lasso model to the data with the coordinate descent algorithm.

        Parameters
        ----------
        Phi : array, shape [N, M]
              matrix of basis functions evaluated at (all) input data points
        y   : array, shape [N, 1]
              true solution values
        lambda_val : float
              L1 regularization strength (=lambda)
        iterations: int
              total number of iterations of the outer loop (optional)
        weights_init: array, shape [M,1]
            array with initial weight values (optional)
        tol: float
            sets the tolerance for smallest possible values of the weights
            such that they do not get converted to zero if they are larger
            than this tolerance (optional)

        """
        N, M = Phi.shape
        if weights_init is not None:
            self.weights = weights_init

        else:
            self.weights = np.zeros([M, 1])
        for i in range(iterations):
            ##############
            ##############

            # TODO

            ##############
            ##############
            self.weights[np.abs(self.weights) <= tol] = 0

    def predict(self, Phi):
        """Generate predictions for the given samples.

        Parameters
        ----------
        Phi : array, shape [N, M]
            Matrix of basis functions evaluated at (all) input data points

        Returns
        -------
        y_pred : array, shape [N, 1]
            Predicted regression outputs for the input data.

        """
        if self.weights is not None:
            y_pred = Phi @ self.weights
        else:
            y_pred = None

        return y_pred.reshape(-1, 1)
