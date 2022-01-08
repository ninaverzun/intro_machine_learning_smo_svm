"""

Course:     20942 - Intro to computational learning
Institute:  Open University of Israel
Assignment: Maman 13
Name:       Nina Verzun
ID:         304680473
Date:       Jan 8 2021

"""
import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler


class SMOsvm:
    """ Implement the Sequential Minimal Optimization (SMO) algorithm designed by J. C. Platt and described in:
     "Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines", 1998.
     """

    def __init__(self, x_in, y_in, c_in, kernel, alphas, b, errors):
        """ Initializing container used for "Sequential Minimal Optimization" (SMO) SVM
        :param x_in: training data vector
        :param y_in: class label vector
        :param c_in: regularization parameter
        :param kernel:  kernel function
        :param alphas: lagrange multiplier vector
        :param b: scalar bias term
        :param errors: error cache
        """
        self.X = x_in
        self.y = y_in
        self.C = c_in
        self.kernel = kernel
        self.lagrange_alphas = alphas
        self.bias = b
        self.errors_cache = errors
        self._obj = []  # record of svm objective function value
        self.m = len(self.X)  # store size of training set

    def svm_dual_function(self, alphas, x_train):
        """
        Returns the dual SVM objective function to be minimised during the training
        """
        return np.sum(alphas) - 0.5 * np.sum(
            (self.y[:, None] * self.y[None, :]) * self.kernel(x_train, x_train) * (alphas[:, None] * alphas[None, :]))

    def decision_function(self, x_train, x_test):
        """Applies the SVM decision function to the input feature vectors in `x_test`."""
        # result = np.matmul((self.lagrange_alphas * self.y), self.kernel(x_train, x_test)) - self.bias
        result = ((self.lagrange_alphas * self.y) @ self.kernel(x_train, x_test)) - self.bias
        return result

    def optimize_pair_step(self, i1, i2):
        # Skip if chosen alphas are the same
        if i1 == i2:
            return 0, self

        global alpha_tolerance

        alph1 = self.lagrange_alphas[i1]
        alph2 = self.lagrange_alphas[i2]
        target1 = self.y[i1]
        target2 = self.y[i2]
        error1 = self.errors[i1]
        error2 = self.errors[i2]
        s = target1 * target2

        # Compute low_bound & high_bound, the bounds on new possible alpha values
        if target1 != target2:
            low_bound = max(0, alph2 - alph1)
            high_bound = min(self.C, self.C + alph2 - alph1)
        elif target1 == target2:
            low_bound = max(0, alph1 + alph2 - self.C)
            high_bound = min(self.C, alph1 + alph2)
        if low_bound == high_bound:
            return 0, self

        # Compute kernel & 2nd derivative eta
        k11 = self.kernel(self.X[i1], self.X[i1])
        k12 = self.kernel(self.X[i1], self.X[i2])
        k22 = self.kernel(self.X[i2], self.X[i2])
        eta = 2 * k12 - k11 - k22

        # Compute new alpha 2 (a2) if eta is negative
        if (eta < 0):
            a2 = alph2 - target2 * (error1 - error2) / eta
            # Clip a2 based on bounds low_bound & high_bound
            if low_bound < a2 < high_bound:
                a2 = a2
            elif (a2 <= low_bound):
                a2 = low_bound
            elif (a2 >= high_bound):
                a2 = high_bound

        # If eta is non-negative, move new a2 to bound with greater objective function value
        else:
            alphas_adj = self.lagrange_alphas.copy()
            alphas_adj[i2] = low_bound
            # objective function output with a2 = low_bound
            Lobj = self.svm_dual_function(alphas_adj, self.X)
            alphas_adj[i2] = high_bound
            # objective function output with a2 = high_bound
            Hobj = self.svm_dual_function(alphas_adj, self.X)
            if Lobj > (Hobj + alpha_tolerance):
                a2 = low_bound
            elif Lobj < (Hobj - alpha_tolerance):
                a2 = high_bound
            else:
                a2 = alph2

        # Push a2 to 0 or C if very close
        if a2 < 1e-8:
            a2 = 0.0
        elif a2 > (self.C - 1e-8):
            a2 = self.C

        # If examples can't be optimized within epsilon (eps), skip this pair
        if (np.abs(a2 - alph2) < alpha_tolerance * (a2 + alph2 + alpha_tolerance)):
            return 0, self

        # Calculate new alpha 1 (a1)
        a1 = alph1 + s * (alph2 - a2)

        # Update threshold b to reflect newly calculated alphas
        # Calculate both possible thresholds
        b1 = error1 + target1 * (a1 - alph1) * k11 + target2 * (a2 - alph2) * k12 + self.bias
        b2 = error2 + target1 * (a1 - alph1) * k12 + target2 * (a2 - alph2) * k22 + self.bias

        # Set new threshold based on if a1 or a2 is bound by low_bound and/or high_bound
        global global_c
        if 0 < a1 and a1 < global_c:
            b_new = b1
        elif 0 < a2 and a2 < global_c:
            b_new = b2
        # Average thresholds if both are bound
        else:
            b_new = (b1 + b2) * 0.5

        # Update model object with new alphas & threshold
        self.lagrange_alphas[i1] = a1
        self.lagrange_alphas[i2] = a2

        # Update error cache
        # Error cache for optimized alphas is set to 0 if they're unbound
        for index, alph in zip([i1, i2], [a1, a2]):
            if 0.0 < alph < self.C:
                self.errors[index] = 0.0

        # Set non-optimized errors based on equation 12.11 in Platt's book
        non_opt = [n for n in range(self.m) if (n != i1 and n != i2)]
        self.errors[non_opt] = self.errors[non_opt] + target1 * (a1 - alph1) * self.kernel(self.X[i1],
                                                                                      self.X[non_opt]) + target2 * (
                                       a2 - alph2) * self.kernel(self.X[i2], self.X[non_opt]) + self.bias - b_new

        # Update model threshold
        self.bias = b_new

        return 1, self

    def examine_example(self, i2, tolerance):
        y2 = self.y[i2]
        alph2 = self.lagrange_alphas[i2]
        E2 = self.errors[i2]
        r2 = E2 * y2

        # Proceed if error is within specified tolerance (tol)
        if ((r2 < - tolerance and alph2 < self.C) or (r2 > tolerance and alph2 > 0)):

            if len(self.lagrange_alphas[(self.lagrange_alphas != 0) & (self.lagrange_alphas != self.C)]) > 1:
                # Use 2nd choice heuristic is choose max difference in error
                if self.errors[i2] > 0:
                    i1 = np.argmin(self.errors)
                elif self.errors[i2] <= 0:
                    i1 = np.argmax(self.errors)
                step_result, res_model = self.optimize_pair_step(i1, i2)
                if step_result:
                    return 1, res_model

            # Loop through non-zero and non-C lagrange_alphas, starting at a random point
            for i1 in np.roll(np.where((self.lagrange_alphas != 0) & (self.lagrange_alphas != self.C))[0],
                              np.random.choice(np.arange(self.m))):
                step_result, res_model = self.optimize_pair_step(i1, i2)
                if step_result:
                    return 1, res_model

            # loop through all lagrange_alphas, starting at a random point
            for i1 in np.roll(np.arange(self.m), np.random.choice(np.arange(self.m))):
                step_result, res_model = self.optimize_pair_step(i1, i2)
                if step_result:
                    return 1, res_model

        return 0, self

    def train(self, tolerance):

        numChanged = 0
        examineAll = 1

        while (numChanged > 0) or (examineAll):
            numChanged = 0
            if examineAll:
                # loop over all training examples
                for i in range(self.lagrange_alphas.shape[0]):
                    examine_result, model_res = self.examine_example(i, tolerance)
                    numChanged += examine_result
                    if examine_result:
                        obj_result = self.svm_dual_function(self.lagrange_alphas, self.X)
                        self._obj.append(obj_result)
            else:
                # loop over examples where alphas are not already at their limits
                for i in np.where((self.lagrange_alphas != 0) & (self.lagrange_alphas != self.C))[0]:
                    examine_result, model_res = self.examine_example(i, tolerance)
                    numChanged += examine_result
                    if examine_result:
                        obj_result = self.svm_dual_function(self.lagrange_alphas, self.X)
                        self._obj.append(obj_result)
            if examineAll == 1:
                examineAll = 0
            elif numChanged == 0:
                examineAll = 1

        return model_res


def linear_kernel(x, y, c=1):
    """Calculates the linear combination kernel function (similar to K(x,y)=(xT+c)^q in question 2 section d with q=1"""
    # return np.matmul(x, y.T) + c
    return x @ y.T + c


def radial_basis_function_kernel(x, y, sigma=1):
    """Returns the gaussian similarity of arrays `x` and `y` with
        kernel width parameter `sigma` (set to 1 by default)."""

    if np.ndim(x) == 1 and np.ndim(y) == 1:
        result = np.exp(- (np.linalg.norm(x - y, 2)) ** 2 / (2 * sigma ** 2))
    elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
        result = np.exp(- (np.linalg.norm(x - y, 2, axis=1) ** 2) / (2 * sigma ** 2))
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        result = np.exp(- (np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2) / (2 * sigma ** 2))
    return result


def plot_decision_boundary(model_in, ax, resolution=100, colors=('b', 'k', 'r'), levels=(-1, 0, 1)):
    """Plots the model's decision boundary on the input axes object.
    Range of decision boundary grid is determined by the training data.
    Returns decision boundary grid and axes object (`grid`, `ax`)."""

    # Generate coordinate grid of shape [resolution x resolution]
    # and evaluate the model over the entire space
    xrange = np.linspace(model_in.X[:, 0].min(), model_in.X[:, 0].max(), resolution)
    yrange = np.linspace(model_in.X[:, 1].min(), model_in.X[:, 1].max(), resolution)
    grid = [[model_in.decision_function(model_in.X, np.array([xr, yr])) for xr in xrange] for yr in yrange]
    grid = np.array(grid).reshape(len(xrange), len(yrange))

    # Plot decision contours using grid and
    # make a scatter plot of training data
    ax.contour(xrange, yrange, grid, levels=levels, linewidths=(1, 1, 1), linestyles=('--', '-', '--'), colors=colors)
    ax.scatter(model_in.X[:, 0], model_in.X[:, 1], c=model_in.y, cmap=plt.cm.viridis, lw=0, alpha=0.25)

    # Plot support vectors (non-zero lagrange_alphas)
    # as circled points (linewidth > 0)
    mask = np.round(model_in.lagrange_alphas, decimals=2) != 0.0
    ax.scatter(model_in.X[mask, 0], model_in.X[mask, 1], c=model_in.y[mask], cmap=plt.cm.viridis, lw=1, edgecolors='k')

    return grid, ax


def test_dimensions():
    x_len, y_len = 5, 10
    actual_dim_linear_k = linear_kernel(np.random.rand(x_len, 1), np.random.rand(y_len, 1)).shape
    expected_dim = (x_len, y_len)
    print("Actual shape linear kernel = " + str(actual_dim_linear_k))
    print("Expected shape linear kernel= " + str(expected_dim))

    resulting_dim_rbf = radial_basis_function_kernel(np.random.rand(x_len, 1), np.random.rand(y_len, 1)).shape
    print("result shape RBF= " + str(resulting_dim_rbf))
    print("expected shape RBF= " + str(expected_dim))


def test_linear_model():
    # Test linear data
    x_train, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train, y)
    y[y == 0] = -1

    # Set model parameters and initial values
    m = len(x_train_scaled)
    initial_alphas = np.zeros(m)
    initial_b = 0.0

    # Instantiate model
    linear_model = SMOsvm(x_train_scaled, y, global_c, linear_kernel, initial_alphas, initial_b, np.zeros(m))

    # Initialize error cache
    initial_error = linear_model.decision_function(linear_model.X, linear_model.X) - linear_model.y
    linear_model.errors = initial_error
    np.random.seed(0)
    output = linear_model.train(error_tolerance)
    fig, ax = plt.subplots()
    grid, ax = plot_decision_boundary(output, ax)
    # grid, ax = plot_decision_boundary(linear_model, ax)
    plt.show()


def test_rbf():
    X_train, y = make_circles(n_samples=500, noise=0.1, factor=0.1, random_state=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train, y)
    y[y == 0] = -1
    # Set model_rbf parameters and initial values

    m = len(X_train_scaled)
    initial_alphas = np.zeros(m)
    initial_b = 0.0

    # Instantiate model_rbf
    model_rbf = SMOsvm(X_train_scaled, y, global_c, radial_basis_function_kernel, initial_alphas, initial_b,
                       np.zeros(m))

    # Initialize error cache
    initial_error = model_rbf.decision_function(model_rbf.X, model_rbf.X) - model_rbf.y
    model_rbf.errors = initial_error
    output = model_rbf.train(error_tolerance)
    fig, ax = plt.subplots()
    grid, ax = plot_decision_boundary(output, ax)
    plt.show()


def take_step(i1, i2, model):
    # Skip if chosen alphas are the same
    if i1 == i2:
        return 0, model

    alph1 = model.alphas[i1]
    alph2 = model.alphas[i2]
    y1 = model.y[i1]
    y2 = model.y[i2]
    E1 = model.errors[i1]
    E2 = model.errors[i2]
    s = y1 * y2

    # Compute L & H, the bounds on new possible alpha values
    if (y1 != y2):
        L = max(0, alph2 - alph1)
        H = min(model.C, model.C + alph2 - alph1)
    elif (y1 == y2):
        L = max(0, alph1 + alph2 - model.C)
        H = min(model.C, alph1 + alph2)
    if (L == H):
        return 0, model

    # Compute kernel & 2nd derivative eta
    k11 = model.kernel(model.X[i1], model.X[i1])
    k12 = model.kernel(model.X[i1], model.X[i2])
    k22 = model.kernel(model.X[i2], model.X[i2])
    eta = 2 * k12 - k11 - k22

    # Compute new alpha 2 (a2) if eta is negative
    if (eta < 0):
        a2 = alph2 - y2 * (E1 - E2) / eta
        # Clip a2 based on bounds L & H
        if L < a2 < H:
            a2 = a2
        elif (a2 <= L):
            a2 = L
        elif (a2 >= H):
            a2 = H

    # If eta is non-negative, move new a2 to bound with greater objective function value
    else:
        alphas_adj = model.alphas.copy()
        alphas_adj[i2] = L
        # objective function output with a2 = L
        Lobj = model.svm_dual_function(alphas_adj, model.y, model.kernel, model.X)
        alphas_adj[i2] = H
        # objective function output with a2 = H
        Hobj = model.svm_dual_function(alphas_adj, model.y, model.kernel, model.X)

        eps = alpha_tolerance
        if Lobj > (Hobj + eps):
            a2 = L
        elif Lobj < (Hobj - eps):
            a2 = H
        else:
            a2 = alph2

    # Push a2 to 0 or C if very close
    if a2 < 1e-8:
        a2 = 0.0
    elif a2 > (model.C - 1e-8):
        a2 = model.C

    # If examples can't be optimized within epsilon (eps), skip this pair
    if (np.abs(a2 - alph2) < eps * (a2 + alph2 + eps)):
        return 0, model

    # Calculate new alpha 1 (a1)
    a1 = alph1 + s * (alph2 - a2)

    # Update threshold b to reflect newly calculated alphas
    # Calculate both possible thresholds
    b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + model.b
    b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + model.b

    # Set new threshold based on if a1 or a2 is bound by L and/or H
    if 0 < a1 and a1 < model.C:
        b_new = b1
    elif 0 < a2 and a2 < model.C:
        b_new = b2
    # Average thresholds if both are bound
    else:
        b_new = (b1 + b2) * 0.5

    # Update model object with new alphas & threshold
    model.alphas[i1] = a1
    model.alphas[i2] = a2

    # Update error cache
    # Error cache for optimized alphas is set to 0 if they're unbound
    for index, alph in zip([i1, i2], [a1, a2]):
        if 0.0 < alph < model.C:
            model.errors[index] = 0.0

    # Set non-optimized errors based on equation 12.11 in Platt's book
    non_opt = [n for n in range(model.m) if (n != i1 and n != i2)]
    model.errors[non_opt] = model.errors[non_opt] + y1 * (a1 - alph1) * model.kernel(model.X[i1],
                                                                                     model.X[non_opt]) + y2 * (
                                        a2 - alph2) * model.kernel(model.X[i2], model.X[non_opt]) + model.b - b_new

    # Update model threshold
    model.b = b_new

    return 1, model


# Set tolerances
error_tolerance = 0.01  # error tolerance
alpha_tolerance = 0.01  # alpha tolerance
global_c = 1000.0

# test_dimensions()
test_linear_model()

global_c = 1.0
# test_rbf()
