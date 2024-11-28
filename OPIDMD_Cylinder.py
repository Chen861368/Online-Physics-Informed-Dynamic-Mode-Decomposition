# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.sparse.linalg import spsolve
from visualization_functions import visualize_matrix,compute_and_plot_singular_values_scatter, plot_frobenius_diff,visualize_eigenvalues
from sklearn.metrics import mean_squared_error, r2_score
from pydmd import PiDMD
from matplotlib.ticker import MaxNLocator
from scipy.sparse import diags
np.random.seed(0)
from scipy.io import loadmat
from numba import cuda
from matplotlib.colors import ListedColormap
from scipy.linalg import eig
from scipy.sparse.linalg import eigs
from matplotlib.colors import LinearSegmentedColormap



# Function to reduce columns in VORTALL from start_col to end_col
def reduce_vortall_columns(VORTALL, nx, ny, start_row, end_row, start_col, end_col):
    """
    Reduces the number of columns in each time snapshot of VORTALL,
    starting from `start_col` and keeping up to `end_col` columns.

    Parameters:
        VORTALL (ndarray): Original matrix of shape (nx*ny, time_steps).
        nx (int): Number of rows in each time snapshot.
        ny (int): Number of columns in each time snapshot.
        start_row (int): Starting row index for reduction (0-based).
        end_row (int): Ending row index (exclusive) for the reduction.
        start_col (int): Starting column index for reduction (0-based).
        end_col (int): Ending column index (exclusive) for the reduction.

    Returns:
        VORTALL_reduced (ndarray): Reduced matrix of shape ((end_row-start_row)*(end_col-start_col), time_steps).
    """
    # Ensure the requested range is within bounds
    if end_col > ny or start_col < 0 or start_col >= end_col:
        raise ValueError("Invalid start_col and end_col range for column reduction")
    if end_row > nx or start_row < 0 or start_row >= end_row:
        raise ValueError("Invalid start_row and end_row range for row reduction")

    # Calculate the number of rows and columns in the reduced matrix
    num_rows = end_row - start_row
    num_columns = end_col - start_col
    reduced_size = num_rows * num_columns

    # Preallocate new VORTALL_reduced matrix
    VORTALL_reduced = np.zeros((reduced_size, VORTALL.shape[1]))

    # Perform dimensionality reduction for each time snapshot
    for i in range(VORTALL.shape[1]):
        # Reshape column vector into (nx, ny) matrix
        temp = VORTALL[:, i].reshape((nx, ny), order='F')
        # Select the range of rows and columns from `start_row` to `end_row` and `start_col` to `end_col`
        temp_reduced = temp[start_row:end_row, start_col:end_col]
        # Flatten the reduced matrix into a column vector and store in new matrix
        VORTALL_reduced[:, i] = temp_reduced.flatten(order='F')

    return VORTALL_reduced


def compute_dmd_matrix(displacements, rank=None):
    """
    Compute the Dynamic Mode Decomposition (DMD) matrix A with an option to use truncated SVD.

    Parameters:
    - displacements: numpy.ndarray, a matrix containing the displacements of each mass at each time step.
    - rank: int, optional, the rank for the truncated SVD.

    Returns:
    - A: numpy.ndarray, the approximated system matrix describing the dynamics.
    """
    # Split the displacement data into X and Y matrices
    X = displacements[:, :-1]
    Y = displacements[:, 1:]

    # Perform the Singular Value Decomposition (SVD) of X
    U, Sigma, Vh = np.linalg.svd(X, full_matrices=False)

    # If a rank is specified, truncate the SVD results
    if rank is not None and rank > 0:
        U = U[:, :rank]
        Sigma = Sigma[:rank]
        Vh = Vh[:rank, :]

    # Construct the diagonal matrix for the pseudo-inverse
    Sigma_inv = np.linalg.inv(np.diag(Sigma))

    # Compute the DMD matrix A
    A = Y @ Vh.T @ Sigma_inv @ U.T

    return A

def projection_onto_symmetric(A):
    """
    Projects the matrix A onto the space of symmetric matrices.
    This is done by averaging A and its transpose to ensure the resulting matrix is symmetric.

    Parameters:
        A (ndarray): The input matrix to be projected.

    Returns:
        ndarray: The symmetric matrix obtained by projecting A onto the space of symmetric matrices.
    """
    return (A + A.T) / 2


def gradient_descent_update_symmetric(A_k, X_k_new, Y_k_new, learning_rate):
    """
    Performs a single step of online gradient descent for matrix regression with a symmetric matrix constraint.
    It computes the gradient of the loss function with respect to A_k, updates A_k by moving in the direction of 
    the negative gradient, and then projects A_k onto the space of symmetric matrices.

    Parameters:
        A_k (ndarray): The current estimate of the matrix at step k.
        X_k_new (ndarray): The input data matrix at step k.
        Y_k_new (ndarray): The observation matrix at step k.
        learning_rate (float): The step size for the gradient descent.

    Returns:
        ndarray: The updated symmetric matrix A_k after applying the gradient descent update and projecting it onto the 
        space of symmetric matrices.
    """
    # Compute the gradient of the loss function with respect to A_k
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T 
    # Update A_k in the direction of the negative gradient, scaled by the learning rate
    A_k -= learning_rate * grad
    # Apply the symmetric matrix constraint by projecting A_k onto the space of symmetric matrices
    A_k = projection_onto_symmetric(A_k)
    return A_k


def evaluate_predictions(actual, predicted):
    """
    Evaluate the prediction quality of a model using MSE and R^2 metrics.

    Parameters:
    - actual (np.ndarray): The actual data.
    - predicted (np.ndarray): The predicted data.

    Returns:
    - mse (float): Mean Squared Error of the predictions.
    - r2 (float): Coefficient of Determination (R^2) of the predictions.
    """
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    return mse, r2


class OnlineDMD:

    def __init__(self, n: int, weighting: float = 0.9) -> None:
        """Creat an object for online DMD
        Usage: odmd = OnlineDMD(n, weighting)

        Args:
            n (int): state dimension x(t) as in  z(t) = f(z(t-1)) or y(t) = f(t, x(t))
            weighting (float, optional): exponential weighting factor
                smaller value allows more adpative learning, 
                but too small weighting may result in model identification instability (relies only on limited recent snapshots). 
                Defaults to 0.9.
        """
        assert isinstance(n, int) and n >= 1
        weighting = float(weighting)
        assert weighting > 0 and weighting <= 1

        self.n = n
        self.weighting = weighting
        self.timestep = 0
        self.A = np.zeros([n, n])
        self._P = np.zeros([n, n])
        # initialize
        self._initialize()
        self.ready = False

    def _initialize(self) -> None:
        """Initialize online DMD with epsilon small (1e-15) ghost snapshot pairs before t=0"""
        epsilon = 1e-15
        alpha = 1.0 / epsilon
        self.A = np.random.randn(self.n, self.n)
        self._P = alpha * np.identity(self.n)  # inverse of cov(X)


    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Update the DMD computation with a new pair of snapshots (x,y)
        Here, if the (discrete-time) dynamics are given by z(t) = f(z(t-1)),
        then (x,y) should be measurements correponding to consecutive states
        z(t-1) and z(t).
        Usage: odmd.update(x, y)

        Args:
            x (np.ndarray): 1D array, shape (n, ), x(t) as in y(t) = f(t, x(t))
            y (np.ndarray): 1D array, shape (n, ), y(t) as in y(t) = f(t, x(t))
        """
        assert x is not None and y is not None
        x, y = np.array(x), np.array(y)
        assert np.array(x).shape == np.array(y).shape
        assert np.array(x).shape[0] == self.n

        # compute P*x matrix vector product beforehand
        Px = self._P.dot(x)
        # compute gamma
        gamma = 1.0 / (1 + x.T.dot(Px))
        # update A
        self.A += np.outer(gamma * (y - self.A.dot(x)), Px)
        # update P, group Px*Px' to ensure positive definite
        self._P = (self._P - gamma * np.outer(Px, Px)) / self.weighting
        # ensure P is SPD by taking its symmetric part
        self._P = (self._P + self._P.T) / 2

        # time step + 1
        self.timestep += 1

        if self.timestep >= 2 * self.n:
            self.ready = True
        return self.A


def run_online_dmd_update(combined_matrix_noise, D, iterations):
    """
    Run the online DMD update process.
    
    Parameters:
    - combined_matrix_noise: Matrix containing noise, where each column represents a time step.
    - D: The target matrix to compare against.
    - iterations: Number of iterations to run the DMD update.
    
    Returns:
    - odmd.A: Updated DMD matrix after all iterations.
    - frobenius_diffs: List of Frobenius norm differences between the updated DMD matrix and D at each iteration.
    """
    n = combined_matrix_noise.shape[0]  # Ensure n matches the number of rows in combined_matrix_noise
    odmd = OnlineDMD(n, 1)  # Initialize OnlineDMD with the correct value of n
    frobenius_diffs = []

    for idx in range(iterations):
        # Extract x_k and flatten it
        x_k = combined_matrix_noise[:, idx:idx+1].flatten()
        # Extract y_k and flatten it
        y_k = combined_matrix_noise[:, idx+1:idx+2].flatten()

        # Update DMD matrix using flattened x_k and y_k
        A_k = odmd.update(x_k, y_k)
        # frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        # frobenius_diffs.append(frobenius_diff)
        print(idx)

    return odmd.A, frobenius_diffs

def run_online_gradient_descent_l1(combined_matrix_noise, A_k,lambda_reg, D,learning_rate, iterations):
    """
    Run online gradient descent using combined_matrix_noise data to update the circulant matrix A_k column by column.
    
    Parameters:
    - combined_matrix_noise: Matrix containing noise data, where each column represents a time step.
    - learning_rate: The learning rate for gradient descent.
    - D: The target matrix to compare against.
    - iterations: Number of iterations to run the gradient descent.
    
    Returns:
    - A_k: Updated circulant matrix after gradient descent.
    - frobenius_diffs: List of Frobenius norm differences between the updated A_k and D at each iteration.
    """

    frobenius_diffs = []

    # Use the number of columns in combined_matrix_noise as the number of iterations
    for idx in range(iterations):
        x_k = combined_matrix_noise[:, idx:idx+1]  # Extract one column as x_k
        y_k = combined_matrix_noise[:, idx+1:idx+2]  # Extract the next column as y_k
        A_k = gradient_descent_update_l1(A_k, x_k, y_k, lambda_reg, learning_rate)  # Update A_k
        print(idx)
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)
    return A_k, frobenius_diffs

def run_online_gradient_descent_symmetric(combined_matrix_noise, A_k,learning_rate, D, iterations):
    """
    Run online gradient descent using combined_matrix_noise data to update the circulant matrix A_k column by column.
    
    Parameters:
    - combined_matrix_noise: Matrix containing noise data, where each column represents a time step.
    - learning_rate: The learning rate for gradient descent.
    - D: The target matrix to compare against.
    - iterations: Number of iterations to run the gradient descent.
    
    Returns:
    - A_k: Updated circulant matrix after gradient descent.
    - frobenius_diffs: List of Frobenius norm differences between the updated A_k and D at each iteration.
    """

    frobenius_diffs = []

    # Use the number of columns in combined_matrix_noise as the number of iterations
    for idx in range(iterations):
        x_k = combined_matrix_noise[:, idx:idx+1]  # Extract one column as x_k
        y_k = combined_matrix_noise[:, idx+1:idx+2]  # Extract the next column as y_k
        A_k = gradient_descent_update_symmetric(A_k, x_k, y_k, learning_rate)  # Update A_k
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)
        print(idx)

    return A_k, frobenius_diffs

def run_online_gradient_descent_low_rank(combined_matrix_noise, A_k,lambda_reg,learning_rate, D, iterations):
    """
    Run online gradient descent using combined_matrix_noise data to update the circulant matrix A_k column by column.
    
    Parameters:
    - combined_matrix_noise: Matrix containing noise data, where each column represents a time step.
    - learning_rate: The learning rate for gradient descent.
    - D: The target matrix to compare against.
    - iterations: Number of iterations to run the gradient descent.
    
    Returns:
    - A_k: Updated circulant matrix after gradient descent.
    - frobenius_diffs: List of Frobenius norm differences between the updated A_k and D at each iteration.
    """
    frobenius_diffs = []

    # Use the number of columns in combined_matrix_noise as the number of iterations
    for idx in range(iterations):
        x_k = combined_matrix_noise[:, idx:idx+1]  # Extract one column as x_k
        y_k = combined_matrix_noise[:, idx+1:idx+2]  # Extract the next column as y_k
        A_k = gradient_descent_update_unclear_norm(A_k, x_k, y_k, lambda_reg, learning_rate)  # Update A_k
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)
        print(idx)

    return A_k, frobenius_diffs

def run_online_gradient_descent_l2(combined_matrix_noise, A_k,lambda_reg,learning_rate, D, iterations):
    """
    Run online gradient descent using combined_matrix_noise data to update the circulant matrix A_k column by column.
    
    Parameters:
    - combined_matrix_noise: Matrix containing noise data, where each column represents a time step.
    - learning_rate: The learning rate for gradient descent.
    - D: The target matrix to compare against.
    - iterations: Number of iterations to run the gradient descent.
    
    Returns:
    - A_k: Updated circulant matrix after gradient descent.
    - frobenius_diffs: List of Frobenius norm differences between the updated A_k and D at each iteration.
    """
    frobenius_diffs = []

    # Use the number of columns in combined_matrix_noise as the number of iterations
    for idx in range(iterations):
        x_k = combined_matrix_noise[:, idx:idx+1]  # Extract one column as x_k
        y_k = combined_matrix_noise[:, idx+1:idx+2]  # Extract the next column as y_k
        A_k = gradient_descent_update_l2(A_k, x_k, y_k, lambda_reg, learning_rate)  # Update A_k
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)
        print(idx)

    return A_k, frobenius_diffs



def run_online_gradient_descent_operator_norm(combined_matrix_noise, A_k,lambda_reg,learning_rate, D, iterations):
    """
    Run online gradient descent using combined_matrix_noise data to update the circulant matrix A_k column by column.
    
    Parameters:
    - combined_matrix_noise: Matrix containing noise data, where each column represents a time step.
    - learning_rate: The learning rate for gradient descent.
    - D: The target matrix to compare against.
    - iterations: Number of iterations to run the gradient descent.
    
    Returns:
    - A_k: Updated circulant matrix after gradient descent.
    - frobenius_diffs: List of Frobenius norm differences between the updated A_k and D at each iteration.
    """
    frobenius_diffs = []

    # Use the number of columns in combined_matrix_noise as the number of iterations
    for idx in range(iterations):
        x_k = combined_matrix_noise[:, idx:idx+1]  # Extract one column as x_k
        y_k = combined_matrix_noise[:, idx+1:idx+2]  # Extract the next column as y_k
        A_k = gradient_descent_update_operator_norm(A_k, x_k, y_k, lambda_reg, learning_rate)  # Update A_k
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)
        np.save("A_grad_operator.npy", A_k) 
        print(idx)

    return A_k, frobenius_diffs





def gradient_descent_update(A_k, X_k_col, Y_k_col, learning_rate):
    # 计算梯度
    gradient = 2 * (A_k @ X_k_col - Y_k_col) @ X_k_col.T 

    # 更新 A_k
    A_k -= learning_rate * gradient
    return A_k


def run_online_gradient_descent(combined_matrix_noise,A_k,learning_rate, iterations):
    """
    Run online gradient descent using combined_matrix_noise data to update the circulant matrix A_k column by column.
    
    Parameters:
    - combined_matrix_noise: Matrix containing noise data, where each column represents a time step.
    - learning_rate: The learning rate for gradient descent.
    - D: The target matrix to compare against.
    - iterations: Number of iterations to run the gradient descent.
    
    Returns:
    - A_k: Updated circulant matrix after gradient descent.
    - frobenius_diffs: List of Frobenius norm differences between the updated A_k and D at each iteration.
    """
    # Use the number of columns in combined_matrix_noise as the number of iterations
    for idx in range(iterations):
        x_k = combined_matrix_noise[:, idx:idx+1]  # Extract one column as x_k
        y_k = combined_matrix_noise[:, idx+1:idx+2]  # Extract the next column as y_k
        A_k = gradient_descent_update(A_k, x_k, y_k, learning_rate)  # Update A_k
        print(idx)
    return A_k


def add_nonstationary_gaussian_noise(signal, noise_ratio):
    """
    Add non-stationary Gaussian noise to a signal. The noise added to each sample is proportional
    to the magnitude of the signal at that point.

    Parameters:
    - signal (np.ndarray): The original signal.
    - noise_ratio (float): The ratio of the noise amplitude to the signal amplitude.

    Returns:
    - noisy_signal (np.ndarray): Signal with added non-stationary Gaussian noise.
    """
    # Calculate noise standard deviation for each sample
    noise_std_per_sample = np.abs(signal) * noise_ratio

    # Generate non-stationary Gaussian noise
    noise = noise_std_per_sample * np.random.normal(0, 1, signal.shape)

    # Add noise to the original signal
    noisy_signal = signal + noise
    return noisy_signal



def l1_soft_thresholding(A, lambda_):
    """
    Applies the soft thresholding operation element-wise to the matrix A.
    This is commonly used in optimization processes for L1 regularization (Lasso).
    It shrinks the values of A towards zero by the threshold lambda_.

    Parameters:
        A (ndarray): The matrix to which the soft thresholding is applied.
        lambda_ (float): The regularization parameter controlling the strength of the thresholding.

    Returns:
        ndarray: The matrix with soft thresholding applied to each element.
    """
    return np.sign(A) * np.maximum(np.abs(A) - lambda_, 0.0)


def gradient_descent_update_l1(A_k, X_k, Y_k, lambda_reg, learning_rate):
    """
    Performs a single step of online gradient descent for L1-regularized least squares.
    It computes the gradient of the loss function with respect to A_k, 
    applies L1 regularization via soft thresholding, and updates A_k by moving in the 
    negative gradient direction scaled by the learning rate.

    Parameters:
        A_k (ndarray): The current estimate of the matrix at step k.
        X_k (ndarray): The input data matrix at step k.
        Y_k (ndarray): The observation matrix at step k.
        lambda_reg (float): The regularization parameter controlling the strength of the L1 regularization.
        learning_rate (float): The step size for the gradient descent.

    Returns:
        ndarray: The updated matrix A_k after applying the gradient descent update with L1 regularization.
    """
    # Compute the gradient of the loss function with respect to A_k
    grad = -2 * (Y_k - A_k @ X_k) @ X_k.T 
    # Update A_k in the direction of the negative gradient, scaled by the learning rate
    A_k -= learning_rate * grad
    # Apply the soft thresholding operator for L1 regularization
    A_k = l1_soft_thresholding(A_k, lambda_reg)
    return A_k


def proximal_unclear_norm(A, lambda_reg):
    """
    Projects the matrix A onto the matrix space constrained by a specified operator norm.
    This is typically used in optimization problems where we need to enforce a norm constraint.

    Parameters:
        A (ndarray): The matrix to be projected.
        lambda_reg (float): The regularization parameter controlling the strength of the norm constraint.

    Returns:
        ndarray: The matrix A projected onto the space with the operator norm constraint.
    """
    # Perform Singular Value Decomposition (SVD) of A
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # Apply the operator norm constraint by shrinking singular values by lambda_reg
    s = np.maximum(s - lambda_reg, 0)
    # Reconstruct the matrix using the modified singular values
    return U @ np.diag(s) @ Vt


def gradient_descent_update_unclear_norm(A_k, X_k_new, Y_k_new, lambda_reg, learning_rate):
    """
    Performs a single step of online gradient descent for matrix regression with an operator norm constraint.
    It computes the gradient of the loss function with respect to A_k, applies the gradient descent update,
    and then projects A_k onto the space constrained by the operator norm using the proximal operator.

    Parameters:
        A_k (ndarray): The current estimate of the matrix at step k.
        X_k_new (ndarray): The input data matrix at step k.
        Y_k_new (ndarray): The observation matrix at step k.
        lambda_reg (float): The regularization parameter controlling the strength of the operator norm constraint.
        learning_rate (float): The step size for the gradient descent.

    Returns:
        ndarray: The updated matrix A_k after applying the gradient descent update and the operator norm constraint.
    """
    # Compute the gradient of the loss function with respect to A_k
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T
    # Update A_k in the direction of the negative gradient, scaled by the learning rate
    A_k -= learning_rate * grad
    # Apply the operator norm constraint using the proximal operator
    A_k = proximal_unclear_norm(A_k, lambda_reg)
    return A_k






def proximal_operator_norm(A, lambda_reg):
    """
    Project the matrix onto a space with a specified operator norm constraint,
    adjusting only the largest singular value.
    """
    # Perform SVD decomposition of matrix A
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # Apply the operator norm constraint only to the largest singular value
    # s[0] = np.maximum(s[0] - lambda_reg, 0)  # Only adjust the largest singular value
    s[0] = np.maximum(s[0] - lambda_reg, 1)  # Only adjust the largest singular value
    s[-1] = np.minimum(s[-1] + lambda_reg, 1)# Adjust only the smallest singular value
    # Reconstruct the matrix
    return U @ np.diag(s) @ Vt


def gradient_descent_update_operator_norm(A_k, X_k_new, Y_k_new, lambda_reg, learning_rate):
    """
    Performs a single step of online gradient descent with an operator norm proximal gradient.

    Parameters:
        A_k (ndarray): The current estimate of the matrix at step k.
        X_k_new (ndarray): The new data matrix (features) at step k.
        Y_k_new (ndarray): The new observation matrix at step k.
        lambda_reg (float): The regularization parameter controlling the strength of the operator norm constraint.
        learning_rate (float): The step size for gradient descent.

    Returns:
        ndarray: The updated matrix after applying the gradient descent step and the operator norm regularization.
    """
    # Compute the gradient of the Frobenius norm loss function
    grad_fro = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T
    A_k -= learning_rate * grad_fro  # Update A_k in the direction of the negative gradient

    # Apply the proximal operator for operator norm regularization
    A_k = proximal_operator_norm(A_k, lambda_reg)
    
    return A_k

def gradient_descent_update_l2(A_k, X_k_col, Y_k_col, lambda_val, learning_rate):
    """
    Performs a single step of gradient descent for L2 regularized matrix regression.

    Parameters:
        A_k (ndarray): The current estimate of the matrix at step k.
        X_k_col (ndarray): The column vector (data) at step k.
        Y_k_col (ndarray): The observation vector at step k.
        lambda_val (float): The regularization parameter controlling the strength of the L2 regularization.
        learning_rate (float): The step size for gradient descent.

    Returns:
        ndarray: The updated matrix after applying the gradient descent step with L2 regularization.
    """
    # Compute the gradient of the loss function with L2 regularization
    gradient = 2 * (A_k @ X_k_col - Y_k_col) @ X_k_col.T + 2 * lambda_val * A_k

    # Update A_k by subtracting the gradient scaled by the learning rate
    A_k -= learning_rate * gradient
    return A_k


def predict_dmd(A, x_0, k):
    """
    Predicts the future states using Dynamic Mode Decomposition (DMD) from step 1 to step k.

    Parameters:
        A (numpy.ndarray): The state transition matrix.
        x_0 (numpy.ndarray): The initial state vector.
        k (int): The number of time steps to predict.

    Returns:
        numpy.ndarray: The predicted state matrix from step 1 to step k, where each column is a predicted state at a time step.
    """
    # Perform eigenvalue decomposition to obtain eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Project the initial state onto the eigenvector space and reshape it as a column vector
    b = np.linalg.solve(eigenvectors, x_0).reshape(-1, 1)
    
    # Compute the powers of the eigenvalues from step 1 to step k, and transpose for broadcasting
    eigenvalue_powers = np.array([eigenvalues ** step for step in range(1, k + 1)]).T
    
    # Use broadcasting to compute the predicted states
    predicted_states = (eigenvectors @ (eigenvalue_powers * b)).real
    
    return predicted_states


def predict_state(eigenvalues, eigenvectors, x_0, k):
    """
    Predicts the future states using Dynamic Mode Decomposition (DMD) from step 1 to step k.

    Parameters:
        eigenvalues (numpy.ndarray): The eigenvalues obtained from the DMD.
        eigenvectors (numpy.ndarray): The eigenvectors obtained from the DMD.
        x_0 (numpy.ndarray): The initial state vector.
        k (int): The number of time steps to predict.

    Returns:
        numpy.ndarray: The predicted state matrix from step 1 to step k, where each column is a predicted state at a time step.
    """
    
    # Project the initial state onto the eigenvector space and reshape it as a column vector
    b = np.linalg.solve(eigenvectors, x_0).reshape(-1, 1)
    
    # Compute the powers of the eigenvalues from step 1 to step k, and transpose for broadcasting
    eigenvalue_powers = np.array([eigenvalues ** step for step in range(1, k + 1)]).T
    
    # Use broadcasting to compute the predicted states
    predicted_states = (eigenvectors @ (eigenvalue_powers * b)).real
    
    return predicted_states


def predict_state_low_rank(eigenvalues, eigenvectors, x_0, k):
    """
    Predicts the future states using a low-rank DMD model from step 1 to step k.

    Parameters:
        eigenvalues (numpy.ndarray): The r eigenvalues of the low-rank DMD model.
        eigenvectors (numpy.ndarray): The r eigenvectors of the low-rank DMD model.
        x_0 (numpy.ndarray): The initial state vector.
        k (int): The number of time steps to predict.

    Returns:
        numpy.ndarray: The predicted state matrix from step 1 to step k, where each column is a predicted state at a time step.
    """
    
    # Project the initial state onto the low-rank eigenvector space, using the pseudo-inverse for non-square matrices
    b = np.linalg.pinv(eigenvectors) @ x_0
    
    # Compute the powers of the eigenvalues from step 1 to step k
    eigenvalue_powers = np.power(eigenvalues[:, np.newaxis], np.arange(1, k + 1))
    
    # Use broadcasting to compute the predicted states
    predicted_states = (eigenvectors @ (eigenvalue_powers * b[:, np.newaxis])).real
    
    return predicted_states


def run_online_gradient_descent1(combined_matrix_noise,A_k,learning_rate, D, iterations):
    """
    Run online gradient descent using combined_matrix_noise data to update the circulant matrix A_k column by column.
    
    Parameters:
    - combined_matrix_noise: Matrix containing noise data, where each column represents a time step.
    - learning_rate: The learning rate for gradient descent.
    - D: The target matrix to compare against.
    - iterations: Number of iterations to run the gradient descent.
    
    Returns:
    - A_k: Updated circulant matrix after gradient descent.
    - frobenius_diffs: List of Frobenius norm differences between the updated A_k and D at each iteration.
    """
    frobenius_diffs = []

    # Use the number of columns in combined_matrix_noise as the number of iterations
    for idx in range(iterations):
        x_k = combined_matrix_noise[:, idx:idx+1]  # Extract one column as x_k
        y_k = combined_matrix_noise[:, idx+1:idx+2]  # Extract the next column as y_k
        A_k = gradient_descent_update(A_k, x_k, y_k, learning_rate)  # Update A_k
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)
        print(idx)

    return A_k, frobenius_diffs

def plot_cylinder(VORT, ny_reduced, nx_reduced, start_col, end_col, start_row, end_row, x_range=(-2, 8), y_range=(-2, 2), nx=199, ny=449, save_path=None):
    """
    Generates a high-contrast vorticity field plot with clear scientific presentation, 
    including the cylinder position and physical coordinate axes.
    Optionally, saves the plot as a PDF file.

    Parameters:
        VORT (ndarray): The reduced vorticity field with shape (nx_reduced, ny_reduced).
        ny_reduced (int): The reduced number of columns in the y-direction.
        nx_reduced (int): The reduced number of rows in the x-direction.
        start_col, end_col, start_row, end_row: Indices for the start and end columns and rows of the reduced data.
        x_range, y_range: The physical range for the x and y axes.
        nx, ny: The original number of rows and columns of the vorticity field.
        save_path (str): If provided, the plot will be saved as a PDF at this path.
    """
    # Calculate the physical range after reduction
    x_min = x_range[0] + (start_col / (ny - 1)) * (x_range[1] - x_range[0])
    x_max = x_range[0] + (end_col / (ny - 1)) * (x_range[1] - x_range[0])
    y_min = y_range[1] - (end_row / (nx - 1)) * (y_range[1] - y_range[0])
    y_max = y_range[1] - (start_row / (nx - 1)) * (y_range[1] - y_range[0])

    # Set up the figure and resolution
    fig, ax = plt.subplots(dpi=300)
    fig.set_size_inches(6, 3)  # Set size suitable for scientific plots

    # Clip vorticity values and define a high-contrast colormap
    vortmin, vortmax = -5, 5
    VORT = np.clip(VORT, vortmin, vortmax)  # Clip the vorticity values

    # Custom high-contrast colormap
    vivid_colormap = LinearSegmentedColormap.from_list(
        "vivid_high_contrast", [
            (0.0, "#000080"),   # Dark blue
            (0.2, "#104E8B"),   # Medium blue
            (0.35, "#00CED1"),  # Light blue
            (0.5, "#FFFFFF"),   # White
            (0.65, "#FF6347"),  # Tomato red
            (0.8, "#8B0000"),   # Dark red
            (1.0, "#FFD700")    # Gold yellow
        ], N=256
    )

    # Plot the vorticity field
    cax = ax.imshow(VORT, cmap=vivid_colormap, origin='upper', extent=[x_min, x_max, y_min, y_max])
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    # Set axis labels and ticks
    ax.set_xlabel("x", fontsize=12, fontname="Times New Roman")
    ax.set_ylabel("y", fontsize=12, fontname="Times New Roman")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Map the original cylinder position and radius to the reduced physical coordinates
    original_cylinder_center_x = 49
    original_cylinder_center_y = 99
    original_radius = 25

    # Compute the cylinder center position in physical coordinates
    cylinder_center_x = x_range[0] + (original_cylinder_center_x / (ny - 1)) * (x_range[1] - x_range[0])
    cylinder_center_y = y_range[1] - (original_cylinder_center_y / (nx - 1)) * (y_range[1] - y_range[0])

    # Plot the cylinder position
    theta = np.linspace(0, 2 * np.pi, 100)
    cylinder_x = cylinder_center_x + 0.5 * np.sin(theta)  # Cylinder at (-1, 0) with radius 0.5
    cylinder_y = cylinder_center_y + 0.5 * np.cos(theta)
    ax.fill(cylinder_x, cylinder_y, color="gray", alpha=0.6)
    ax.plot(cylinder_x, cylinder_y, 'k', linewidth=0.9)

    # Set the font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    plt.tight_layout()

    # If a path is provided, save the plot as a PDF
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Image saved as {save_path}")

    plt.show()


def compute_dmd(VORTALL_reduced, r=21):
    """
    Computes the Dynamic Mode Decomposition (DMD) of the reduced VORTALL data.

    Parameters:
        VORTALL_reduced (ndarray): The reduced matrix with shape (nx*k, time_steps).
        r (int): The number of modes to retain in the decomposition.

    Returns:
        eigs_vals (ndarray): Eigenvalues of the reduced linear map Atilde.
        W (ndarray): Eigenvectors of the reduced linear map Atilde.
        Phi (ndarray): DMD modes.
    """
    # Define the reduced VORTALL_reduced as X and X2
    X = VORTALL_reduced[:, :-1]
    X2 = VORTALL_reduced[:, 1:]

    # Perform SVD on X
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    S = np.diag(s)

    # Truncate to retain only r modes
    U_r = U[:, :r]
    S_r = S[:r, :r]
    V_r = Vh[:r, :].conj().T

    # Compute Atilde
    Atilde = U_r.T.conj() @ X2 @ V_r @ np.linalg.inv(S_r)

    # Compute eigenvalues and eigenvectors of Atilde
    eigs_vals, W = eig(Atilde)

    # Compute DMD modes (Phi)
    Phi = X2 @ V_r @ np.linalg.inv(S_r) @ W

    return eigs_vals, W, Phi,X2,V_r,S_r,U_r

def compute_top_n_eigenpairs(A, n):
    """
    Compute the top n eigenvalues and eigenvectors of matrix A.
    
    Parameters:
    A (ndarray or sparse matrix): The input matrix (should be square).
    n (int): The number of top eigenvalues and eigenvectors to compute.
    
    Returns:
    tuple: A tuple containing the top n eigenvalues and their corresponding eigenvectors.
    """
    # Compute the top n eigenvalues and eigenvectors of matrix A
    eigenvalues, eigenvectors = eigs(A, k=n)  # Using `eigs` from scipy to compute the eigenpairs
    
    return eigenvalues, eigenvectors  # Return the computed eigenvalues and eigenvectors


def plot_eigenvalues(eigenvalues, save_path=None):
    """
    Plot the eigenvalues on the complex plane with a unit circle for visual analysis.
    
    Parameters:
    eigenvalues (ndarray): An array of eigenvalues, which may be complex.
    save_path (str or None): Path to save the figure as a PDF file. If None, the figure is not saved.
    """
    # Extract the real and imaginary parts of the eigenvalues
    real_parts = np.real(eigenvalues)  # Real part of the eigenvalues
    imag_parts = np.imag(eigenvalues)  # Imaginary part of the eigenvalues

    # Create the figure and axis for plotting
    fig, ax = plt.subplots(dpi=300)  # High resolution for scientific publishing
    fig.set_size_inches(6, 6)  # Square figure to fit the complex plane better
    
    # Scatter plot of the eigenvalues on the complex plane
    ax.scatter(real_parts, imag_parts, color="blue", s=30, edgecolor="black", alpha=0.7, label="Eigenvalues")

    # Draw the real and imaginary axes
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Horizontal line for real axis
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')  # Vertical line for imaginary axis

    # Add a unit circle (radius = 1) to the plot
    unit_circle = plt.Circle((0, 0), 1, color="red", linestyle='--', linewidth=1, fill=False, alpha=0.7, label="Unit Circle")
    ax.add_artist(unit_circle)

    # Set the labels for the axes
    ax.set_xlabel("Real Part", fontsize=12, fontname="Times New Roman")
    ax.set_ylabel("Imaginary Part", fontsize=12, fontname="Times New Roman")

    # Set the plot limits to ensure the unit circle is fully visible
    ax.set_xlim(-1.5, 1.5)  # Real part range
    ax.set_ylim(-1.5, 1.5)  # Imaginary part range

    # Add gridlines to enhance the scientific style of the plot
    ax.grid(which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # Ensure the plot is square, so the unit circle appears correctly
    ax.set_aspect('equal', adjustable='box')

    # Add a legend in the upper right corner with a frame around it
    ax.legend(fontsize=10, loc="upper right", frameon=True, framealpha=0.9, edgecolor="black")

    # Use Times New Roman font for consistency with academic publishing
    plt.rcParams['font.family'] = 'Times New Roman'

    # If a save path is provided, save the plot as a PDF file
    if save_path is not None:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')  # Save the plot as a PDF file
        print(f"Figure saved as PDF at: {save_path}")  # Print the save location

    # Show the plot
    plt.tight_layout()  # Adjust the layout to avoid clipping
    plt.show()  # Display the plot

    return fig  # Return the figure object for further use or modification


# Load the data from CYLINDER_ALL.mat
data = loadmat('CYLINDER_ALL.mat')
VORTALL = data['VORTALL']

# Define original grid size
nx = 199
ny = 449

# Define the starting and ending columns for reduction
start_col = 19  # Starting column (0-based index)
end_col = 180  # Ending column (exclusive)
# end_col = 50  # Ending column (exclusive)

# start_row = 80
start_row = 40
end_row= nx-start_row
# r=5
nx_reduced = end_row - start_row
ny_reduced = end_col - start_col


# Reduce VORTALL to keep only columns from start_col to end_col
u = reduce_vortall_columns(VORTALL, nx, ny,start_row, end_row, start_col, end_col)


# Use a for loop to visualize the first 100 time steps with noise
for t in range(1):
    mode_real = u[:, t].reshape((nx_reduced, ny_reduced), order='F')
    plot_cylinder(mode_real, ny_reduced, nx_reduced, start_col, end_col, start_row, end_row)


# Parameter settings
noise_ratio = 0.25    # Noise ratio to add Gaussian noise to the data
rank = 50             # Rank of the DMD (Dynamic Mode Decomposition) matrix
# learning_rate = 0.000003 # Learning rate for online gradient descent (slower decay)
# learning_rate = 0.0000015 # Learning rate for faster decay in online gradient descent
learning_rate = 0.00000001  # Learning rate for online gradient descent (faster decay)
# learning_rate = 0.00000002 # Learning rate for faster decay in online gradient descent
# learning_rate = 0.0000001  # Learning rate for online gradient descent
# lambda_reg_l1 = 0.000000005  # Regularization term for L1 regularization
lambda_reg_low_rank = 0.000001  # Regularization term for low-rank matrix
lambda_reg_l2 = 0.0001          # Regularization term for L2 regularization
lambda_reg_operator = 0.0001    # Regularization term for operator norm

# Adding nonstationary Gaussian noise to the input data
noisy_u = add_nonstationary_gaussian_noise(u, noise_ratio)

# Display the first 100 time steps using a loop (only 1 time step in this case)
for t in range(1):  
    # Reshape the noisy data for the given time step to match the reduced dimensions
    mode_real = noisy_u[:, t].reshape((nx_reduced, ny_reduced), order='F')
    print(mode_real.shape)  # Print the shape of the reshaped data (mode shape)
    
    # Plot the data and save the figure as a PDF
    plot_cylinder(mode_real, ny_reduced, nx_reduced, start_col, end_col, start_row, end_row)

# Prediction settings
k_steps = 10               # Number of prediction steps ahead
# k_steps = 50             # Alternative: Set to 50 prediction steps
u_0 = u[:, -1 - k_steps]   # Use the state at the time step k+1 (counting backwards) as the initial state
u_train = noisy_u[:, :-1 - k_steps]  # Training data, excluding the last k_steps
u_test = noisy_u[:, -k_steps:]       # Test data used for validation of predictions
iterations = u_train.shape[1] - 1    # Number of iterations for the training loop (based on the training data shape)

# Compute the DMD matrix
eigenvalues, W, eigenvectors, X2, V_r, S_r, U_r = compute_dmd(u_train, rank)  # Compute DMD and extract eigenvalues, eigenvectors, etc.
A = compute_dmd_matrix(u_train, rank)  # Compute the DMD matrix
plot_eigenvalues(eigenvalues)  # Plot the eigenvalues of the DMD matrix

# Predict the states using the low-rank approximation of the DMD matrix
predicted_states_A = predict_state_low_rank(eigenvalues, eigenvectors, u_0, k_steps)

# Evaluate the prediction accuracy using mean squared error and R² score
mse_optimized, r2_A = evaluate_predictions(u_test, predicted_states_A)
print("R² for A:", r2_A)  # Print the R² score of the predictions for the DMD matrix


n = u_train.shape[0]  # Get the number of samples (rows) in the training data
num_rounds = 1  # Set the number of training rounds to 1

# Load the gradient matrix A_grad from a previously saved file
A_grad = np.load("A_grad1.npy")  
# A_grad = np.load("A_grad_l1.npy")  # Uncomment to load a different gradient matrix file
# A_grad = np.load("A_grad_l2.npy")  # Uncomment to load another variant
# A_grad = np.load("A_grad_symmetric.npy")  # Uncomment to load symmetric variant
print('load')  # Indicate that the file has been loaded

# The loop below is commented out, but it would execute multiple rounds of training if enabled
# for round_num in range(num_rounds):  
#     # If it's not the first round, load the gradient matrix from the previous round as the initial value
#     if round_num > 0:
#         A_grad = np.load("A_grad1.npy")  # Load the updated gradient matrix for the next round
#         # A_grad = np.load("A_grad_l1.npy")  # Uncomment to load a different variant for the next round
#         # A_grad = np.load("A_grad_symmetric.npy")  # Uncomment to load symmetric variant for next round
#         # A_grad = np.load("A_grad_l2.npy")  # Uncomment to load a different variant for next round
#         # A_grad = np.load("A_grad_low_rank.npy")  # Uncomment to load another variant

#     # Run the online gradient descent update for the current round
#     # A_grad = run_online_gradient_descent(u_train, A_grad, learning_rate, iterations)
#     A_grad, frobenius_diffs = run_online_gradient_descent1(u_train, A_grad, learning_rate, A, iterations)
#     # A_grad, frobenius_diffs = run_online_gradient_descent_symmetric(u_train, A_grad, learning_rate, A, iterations)
#     # A_grad, frobenius_diffs = run_online_gradient_descent_l1(u_train, A_grad, lambda_reg_l1, A, learning_rate, iterations)
#     # A_grad, frobenius_diffs = run_online_gradient_descent_l2(u_train, A_grad, lambda_reg_l2, learning_rate, A, iterations)
#     # A_grad, frobenius_diffs = run_online_gradient_descent_operator_norm(u_train, A_grad, lambda_reg_operator, learning_rate, A, iterations)
#     # A_grad, frobenius_diffs = run_online_gradient_descent_low_rank(u_train, A_grad, lambda_reg_low_rank, learning_rate, A, iterations)
#     # Plot the Frobenius norm difference curve to visualize the convergence during training
#     plot_frobenius_diff(frobenius_diffs)

#     # Save the updated gradient matrix A_grad to a file for future use
#     # np.save("A_grad1.npy", A_grad)  # Save the updated matrix for later rounds
#     # np.save("A_grad_l1.npy", A_grad)  # Save using a different filename to keep track of variants
#     # np.save("A_grad_l2.npy", A_grad)  # Save using a different filename
#     # np.save("A_grad_symmetric.npy", A_grad)  # Save using a symmetric variant
#     np.save("A_grad_low_rank.npy", A_grad)  # Save the low-rank variant for future use


# OPIDMD
eigenvalues, eigenvectors = compute_top_n_eigenpairs(A_grad, 50)
plot_eigenvalues(eigenvalues)
predicted_states_A_grad = predict_state_low_rank(eigenvalues, eigenvectors, u_0, k_steps)
mse_optimized, r2_A_k_grad = evaluate_predictions(u_test, predicted_states_A_grad)
print("R² for A_k_grad:", r2_A_k_grad)

# pidmd
unitary_matrix_pi = PiDMD(manifold="unitary", compute_A=True).fit(u_train)
eigenvalues, eigenvectors = compute_top_n_eigenpairs(unitary_matrix_pi.A, 50)
plot_eigenvalues(eigenvalues)
predicted_states_A_pi = predict_state_low_rank(eigenvalues, eigenvectors, u_0, k_steps)
mse_optimized, r2_pi = evaluate_predictions(u_test, predicted_states_A_pi)
print("R² for piDMD:", r2_pi)



# online DMD
# online_dmd, frobenius_diffs_odmd = run_online_dmd_update(u_train, A, iterations)
# np.save("A_odmd.npy", online_dmd)   
online_dmd = np.load("A_odmd.npy")
eigenvalues, eigenvectors = compute_top_n_eigenpairs(online_dmd, 50)
plot_eigenvalues(eigenvalues)
predicted_states_odmd= predict_state_low_rank(eigenvalues, eigenvectors, u_0, k_steps)
mse_dmd, r2_odmd = evaluate_predictions(u_test, predicted_states_odmd)
print("R² for A_odmd:", r2_odmd)








































