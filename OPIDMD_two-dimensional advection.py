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
    Projects a matrix onto the space of symmetric matrices.
    """
    return (A + A.T) / 2

def gradient_descent_update_symmetric(A_k, X_k_new, Y_k_new, learning_rate):
    """
    Performs a single update step using online gradient descent 
    and applies a symmetric matrix constraint.
    """
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T  # Compute the gradient
    A_k -= learning_rate * grad  # Update the matrix using the gradient
    A_k = projection_onto_symmetric(A_k)  # Enforce the symmetric matrix constraint
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
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)

    return odmd.A, frobenius_diffs

def run_online_gradient_descent_l1(combined_matrix_noise, lambda_reg, learning_rate, iterations):
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
    n = combined_matrix_noise.shape[0]
    A_k = np.zeros((n, n))  # Initialize A_k as an all-zero matrix
    y_estimates = np.zeros((n, iterations))

    # Use the number of columns in combined_matrix_noise as the number of iterations
    for idx in range(iterations):
        x_k = combined_matrix_noise[:, idx:idx+1]  # Extract one column as x_k
        y_k = combined_matrix_noise[:, idx+1:idx+2]  # Extract the next column as y_k
        A_k = gradient_descent_update_l1(A_k, x_k, y_k, lambda_reg, learning_rate)  # Update A_k
        y_k_pred = A_k @ x_k
        # print(y_k_pred-y_k)
        y_estimates[:, idx] = y_k_pred.flatten()

    return A_k, y_estimates

def run_online_gradient_descent_symmetric(combined_matrix_noise, learning_rate, D, iterations):
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
    n = combined_matrix_noise.shape[0]
    A_k = np.zeros((n, n))  # Initialize A_k as an all-zero matrix
    frobenius_diffs = []

    # Use the number of columns in combined_matrix_noise as the number of iterations
    for idx in range(iterations):
        x_k = combined_matrix_noise[:, idx:idx+1]  # Extract one column as x_k
        y_k = combined_matrix_noise[:, idx+1:idx+2]  # Extract the next column as y_k
        A_k = gradient_descent_update_symmetric(A_k, x_k, y_k, learning_rate)  # Update A_k
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)

    return A_k, frobenius_diffs

def run_online_gradient_descent_low_rank(combined_matrix_noise, lambda_reg,learning_rate, D, iterations):
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
    n = combined_matrix_noise.shape[0]
    A_k = np.zeros((n, n))  # Initialize A_k as an all-zero matrix
    frobenius_diffs = []

    # Use the number of columns in combined_matrix_noise as the number of iterations
    for idx in range(iterations):
        x_k = combined_matrix_noise[:, idx:idx+1]  # Extract one column as x_k
        y_k = combined_matrix_noise[:, idx+1:idx+2]  # Extract the next column as y_k
        A_k = gradient_descent_update_unclear_norm(A_k, x_k, y_k, lambda_reg, learning_rate)  # Update A_k
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)

    return A_k, frobenius_diffs

def run_online_gradient_descent_l2(combined_matrix_noise, lambda_reg,learning_rate, D, iterations):
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
    n = combined_matrix_noise.shape[0]
    A_k = np.zeros((n, n))  # Initialize A_k as an all-zero matrix
    frobenius_diffs = []

    # Use the number of columns in combined_matrix_noise as the number of iterations
    for idx in range(iterations):
        x_k = combined_matrix_noise[:, idx:idx+1]  # Extract one column as x_k
        y_k = combined_matrix_noise[:, idx+1:idx+2]  # Extract the next column as y_k
        A_k = gradient_descent_update_l2(A_k, x_k, y_k, lambda_reg, learning_rate)  # Update A_k
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)

    return A_k, frobenius_diffs

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


def plot_comparison(time, actual_data, piDMD_data, DMD_data, mass_index, name,save_path="C:\\Users\\HIT\\Desktop"):
    """
    Plot a comparison of actual data with predictions from online piDMD and online DMD, using styles akin to Nature or Science publications.
    
    Parameters:
    - time: Array of time steps.
    - actual_data: Actual data array.
    - piDMD_data: Predicted data array from online piDMD.
    - DMD_data: Predicted data array from online DMD.
    - mass_index: Index of the mass for labeling.
    - save_path: Path to save the plot as a PDF.
    """
    # Set style
    sns.set(style="white", context="talk")
    plt.figure(figsize=(12, 8), dpi=300)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    # Plotting with updated colors and line styles
    plt.plot(time, actual_data, label='Actual Data', color='grey', linewidth=2)
    plt.plot(time, piDMD_data, label='OPIDMD', color='blue', linestyle='-.', linewidth=2)
    plt.plot(time, DMD_data, label='Online DMD', color='orange', linestyle=':', linewidth=2.5)

    # Labels and title with updated font sizes
    plt.xlabel('Time (seconds)', fontsize=25)
    plt.ylabel(name, fontsize=25)
    # plt.ylabel('Velocity', fontsize=25)

    # Legend
    legend = plt.legend(fontsize='x-large', loc='upper right', edgecolor='black', frameon=True, fancybox=False)
    legend.get_frame().set_linewidth(1.5)

    # Grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Saving
    filename = f"{save_path}/comparison_Displacement_mass_{mass_index+1}.pdf"
    # filename = f"{save_path}/comparison_velocity_mass_{mass_index+1}.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()  # Display the figure


def l1_soft_thresholding(A, lambda_):
    """
    Applies soft thresholding to each element of matrix A.
    Soft thresholding is used for L1 regularization (Lasso) in optimization processes.
    It shrinks the values of A towards zero by a threshold defined by lambda_.
    """
    return np.sign(A) * np.maximum(np.abs(A) - lambda_, 0.0)

def gradient_descent_update_l1(A_k, X_k, Y_k, lambda_reg, learning_rate):
    """
    Performs a single step of online gradient descent.
    This involves computing the gradient of the loss function with respect to A_k,
    applying L1 regularization, and updating A_k along the negative gradient direction 
    scaled by the learning rate. It also includes the proximal step for L1 regularization.
    """
    # Compute the gradient of the loss function
    grad = -2 * (Y_k - A_k @ X_k) @ X_k.T 
    # Update A_k along the negative gradient direction
    A_k -= learning_rate * grad
    # Apply soft thresholding for L1 regularization
    A_k = l1_soft_thresholding(A_k, lambda_reg)
    return A_k

def proximal_unclear_norm(A, lambda_reg):
    """
    Projects a matrix onto the space of matrices with a specified operator norm constraint.
    """
    # Perform Singular Value Decomposition (SVD) of matrix A
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # Apply the operator norm constraint
    s = np.maximum(s - lambda_reg, 0)
    # Reconstruct the matrix
    return U @ np.diag(s) @ Vt

def gradient_descent_update_unclear_norm(A_k, X_k_new, Y_k_new, lambda_reg, learning_rate):
    """
    Performs a single update step using online gradient descent and applies 
    an operator norm constraint.
    """
    # Compute the gradient
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T
    # Gradient descent step
    A_k -= learning_rate * grad
    # Apply the operator norm constraint
    A_k = proximal_unclear_norm(A_k, lambda_reg)
    return A_k

def predict_dmd(A, x_0, k):
    """
    Predicts future states from step 1 to step k using the DMD (Dynamic Mode Decomposition) method.
    
    Parameters:
    A (numpy.ndarray): State transition matrix.
    x_0 (numpy.ndarray): Initial state vector.
    k (int): Number of time steps to predict.
    
    Returns:
    numpy.ndarray: Predicted state matrix from step 1 to step k, where each column represents 
                   the predicted state at a specific time step.
    """
    # Perform eigendecomposition to get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Project the initial state onto the eigenvector space and reshape to a column vector
    b = np.linalg.solve(eigenvectors, x_0).reshape(-1, 1)
    
    # Compute powers of the eigenvalues for each step from 1 to k, and transpose for broadcasting
    eigenvalue_powers = np.array([eigenvalues ** step for step in range(1, k + 1)]).T
    
    # Compute predicted states using broadcasting
    predicted_states = (eigenvectors @ (eigenvalue_powers * b)).real
    
    return predicted_states


def gradient_descent_update_l2(A_k, X_k_col, Y_k_col, lambda_val, learning_rate):
    """
    Performs a single gradient descent update step with L2 regularization.

    Parameters:
    A_k (numpy.ndarray): Current state of the parameter matrix.
    X_k_col (numpy.ndarray): Column vector representing input data.
    Y_k_col (numpy.ndarray): Column vector representing target data.
    lambda_val (float): Regularization parameter for L2 norm.
    learning_rate (float): Step size for the gradient descent update.

    Returns:
    numpy.ndarray: Updated parameter matrix A_k.
    """
    # Compute the gradient of the loss function with L2 regularization
    gradient = 2 * (A_k @ X_k_col - Y_k_col) @ X_k_col.T + 2 * lambda_val * A_k

    # Update A_k using the gradient and learning rate
    A_k -= learning_rate * gradient
    return A_k


# Define the 2D advection equation solver using an upwind scheme
def solve_advection_2d(Lx, Ly, T, Nx, Ny, Nt, cx, cy, u_initial_func):
    """
    Solves the 2D advection equation using an upwind scheme.

    Parameters:
    - Lx, Ly (float): Length of the spatial domain in the x and y directions.
    - T (float): Total time.
    - Nx, Ny (int): Number of grid points in the x and y directions.
    - Nt (int): Number of time steps.
    - cx, cy (float): Advection velocities in the x and y directions.
    - u_initial_func (function): Initial condition function that takes x, y as inputs and returns the initial field.

    Returns:
    - x, y (array): Spatial grids in the x and y directions.
    - t (array): Time grid.
    - u (3D array): Solution matrix with shape (Nx, Ny, Nt).
    """
    dx = Lx / (Nx - 1)  # Spatial step size in the x direction
    dy = Ly / (Ny - 1)  # Spatial step size in the y direction
    dt = T / (Nt - 1)   # Time step size
    
    # Create spatial and time grids
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    t = np.linspace(0, T, Nt)
    
    # Initialize the solution matrix
    u = np.zeros((Nx, Ny, Nt))
    
    # Set the initial condition at t=0
    X, Y = np.meshgrid(x, y, indexing='ij')  # Create a 2D meshgrid
    u[:, :, 0] = u_initial_func(X, Y)        # Set the initial condition
    
    # Time-stepping loop using an upwind scheme
    for n in range(Nt - 1):
        for i in range(1, Nx):
            for j in range(1, Ny):
                # Upwind scheme for the x direction
                ux = (u[i, j, n] - u[i - 1, j, n]) / dx if cx >= 0 else (u[i + 1, j, n] - u[i, j, n]) / dx
                
                # Upwind scheme for the y direction
                uy = (u[i, j, n] - u[i, j - 1, n]) / dy if cy >= 0 else (u[i, j + 1, n] - u[i, j, n]) / dy
                
                # Update the solution for the next time step
                u[i, j, n + 1] = u[i, j, n] - dt * (cx * ux + cy * uy)
    
    return x, y, t, u
# Function to plot the solution of the 2D advection equation
def plot_solution(x, y, t, u, step=250, save_path=None):
    """
    Plots contour maps of the solution at different time steps. 
    If a save path is provided, saves the plots as PDF files.

    Parameters:
    - x, y (array): Spatial grids in the x and y directions.
    - t (array): Time grid.
    - u (3D array): Solution matrix of shape (Nx, Ny, Nt).
    - step (int): Interval between time steps to plot.
    - save_path (str): If provided, saves the plots as PDF files at the specified path.
    """
    X, Y = np.meshgrid(x, y, indexing='ij')  # Create a 2D spatial grid

    for n in range(0, len(t), step):
        plt.figure(figsize=(8, 6), dpi=300)
        plt.contourf(X, Y, u[:, :, n], levels=100, cmap='viridis')  # Create a filled contour plot
        # plt.title(f'2D Advection at t = {t[n]:.2f}')  # Add a title
        plt.xlabel('x')  # Label for the x-axis
        plt.ylabel('y')  # Label for the y-axis

        # If a save path is provided, save the plot as a PDF file
        if save_path:
            # Create a unique file name using the time step `n`
            # file_path = f"{save_path}/solution_t_{n}.pdf"
            file_path = f"{save_path}/solution_t_{n}_noisy.pdf"
            plt.savefig(file_path, format='pdf', bbox_inches='tight')
            print(f"Plot saved as {file_path}")
        else:
            plt.show()  # If no save path is provided, display the plot



# Step 1: Define parameters for 2D advection
Lx, Ly = 1.0, 1.0  # Spatial domain lengths
T = 1.0            # Total time
Nx, Ny = 15, 15  # Number of grid points in x and y directions
Nt = 5000           # Number of time steps
cx, cy = 0.5, 0.5  # Advection velocities
# cx, cy = 1.0, 1.0  # Advection velocities

# Define the initial condition function
u_initial = lambda X, Y: np.sin(np.pi * X) * np.sin(np.pi * Y)
# u_initial = lambda X, Y: np.sin(3 * np.pi * X) * np.sin(3 * np.pi * Y)
# u_initial = lambda X, Y: np.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2))


# def u_initial_func(X, Y):
#     u = np.sin(np.pi * X) * np.sin(np.pi * Y)
#     u[u < 0.5] = 0  # 将小于0.5的值置为0
#     return u

# u_initial = u_initial_func


# Step 2: Solve the 2D advection equation
x, y, t, u = solve_advection_2d(Lx, Ly, T, Nx, Ny, Nt, cx, cy, u_initial)
# print(u.shape)
plot_solution(x, y, t, u, step=500)


# Parameter settings
noise_ratio = 0.25       # Noise ratio
rank = 10                # Rank of the DMD matrix
learning_rate = 0.00005  # Learning rate for online gradient descent
lambda_reg_l1 = 0.000000004  # Regularization parameter for L1 regularization
# lambda_reg_low_rank = 0.00001  # Alternative low-rank regularization parameter
lambda_reg_low_rank = 0.000001  # Current low-rank regularization parameter
lambda_reg_l2 = 0.0001          # Regularization parameter for L2 regularization

u = u.reshape(Nx * Ny, Nt)  # Reshape solution matrix `u` to 2D form (Nx * Ny, Nt)

# Add non-stationary Gaussian noise
noisy_u = add_nonstationary_gaussian_noise(u, noise_ratio)

# Reshape `noisy_u` back to 3D form (Nx, Ny, Nt)
noisy_u1 = noisy_u.reshape(Nx, Ny, Nt, order='F')

# Plot the noisy solution using `plot_solution`
plot_solution(x, y, t, noisy_u1, step=500)

# Prediction settings
k_steps = 150                  # Number of time steps for prediction
# k_steps = 50                 # Alternative prediction step size
u_0 = u[:, -1 - k_steps]       # Use the (k+1)-th last time step as the initial state
u_train = noisy_u[:, :-1 - k_steps]  # Training data
u_test = noisy_u[:, -k_steps:]       # True data for prediction validation
iterations = u_train.shape[1] - 1    # Number of iterations

# Compute DMD matrices
A = compute_dmd_matrix(u[:, :-1 - k_steps], rank)  # DMD matrix with rank constraint
A_full_rank = compute_dmd_matrix(u, None)         # Full-rank DMD matrix
A_noisy = compute_dmd_matrix(noisy_u, rank)       # DMD matrix with noise and rank constraint
A_noisy_full_rank = compute_dmd_matrix(noisy_u, None)  # Full-rank DMD matrix with noise

# Optimize OPIDMD matrices
A_k_symmetric, frobenius_diffs_opidmd = run_online_gradient_descent_symmetric(u_train , learning_rate, A, iterations)
A_k_l1, frobenius_diffs_opidmd = run_online_gradient_descent_l1(u_train, lambda_reg_l1,learning_rate,iterations)
A_k_l2, frobenius_diffs_opidmd = run_online_gradient_descent_l2(u_train, lambda_reg_l2,learning_rate,A,iterations)
A_grad, frobenius_diffs_opidmd = run_online_gradient_descent_l2(u_train, 0,learning_rate,A,iterations)
A_k_low, frobenius_diffs_opidmd = run_online_gradient_descent_low_rank(u_train, lambda_reg_low_rank ,learning_rate,A, iterations)
online_dmd, frobenius_diffs_odmd = run_online_dmd_update(u_train , A, iterations)


# piDMD
triangular_matrix_pi = PiDMD(manifold="lowertriangular", compute_A=True).fit(u_train)
circulant_matrix_pi = PiDMD(manifold="circulant", compute_A=True).fit(u_train)
diagonal_matrix_pi = PiDMD(manifold="diagonal", compute_A=True).fit(u_train)
symmetric_matrix_pi = PiDMD(manifold="symmetric", compute_A=True).fit(u_train)
unitary_matrix_pi = PiDMD(manifold="unitary", compute_A=True).fit(u_train)

# Perform k-step prediction
predicted_states_symmetric = predict_dmd(A_k_symmetric, u_0, k_steps)
predicted_states_A_k_l1 = predict_dmd(A_k_l1, u_0, k_steps)
predicted_states_A_k_l2 = predict_dmd(A_k_l2, u_0, k_steps)
predicted_states_A_k_low = predict_dmd(A_k_low, u_0, k_steps)
predicted_states_odmd = predict_dmd(online_dmd, u_0, k_steps)
predicted_states_A = predict_dmd(A, u_0, k_steps)
predicted_states_A_noisy = predict_dmd(A_noisy, u_0, k_steps)
predicted_states_A_full_rank = predict_dmd(A_full_rank, u_0, k_steps)
predicted_states_A_noisy_full_rank = predict_dmd(A_noisy_full_rank, u_0, k_steps)
predicted_states_A_grad = predict_dmd(A_grad, u_0, k_steps)
predicted_states_A_symmetric_matrix_pi = predict_dmd(symmetric_matrix_pi.A, u_0, k_steps)
predicted_states_A_circulant_matrix_pi = predict_dmd(circulant_matrix_pi.A, u_0, k_steps)
predicted_states_A_diagonal_matrix_pi = predict_dmd(diagonal_matrix_pi.A, u_0, k_steps)
predicted_states_A_triangular_matrix_pi = predict_dmd(triangular_matrix_pi.A, u_0, k_steps)

# Compute the prediction error
mse_optimized, r2_symmetric = evaluate_predictions(u_test, predicted_states_symmetric)
mse_optimized, r2_A_k_l1 = evaluate_predictions(u_test, predicted_states_A_k_l1)
mse_optimized, r2_A_k_grad = evaluate_predictions(u_test, predicted_states_A_grad)
mse_optimized, r2_A_k_l2 = evaluate_predictions(u_test, predicted_states_A_k_l2)
mse_optimized, r2_A_k_low = evaluate_predictions(u_test, predicted_states_A_k_low)
mse_dmd, r2_odmd = evaluate_predictions(u_test, predicted_states_odmd)
mse_optimized, r2_A = evaluate_predictions(u_test, predicted_states_A)
mse_optimized, r2_A_noisy = evaluate_predictions(u_test, predicted_states_A_noisy)
mse_optimized, r2_A_full_rank = evaluate_predictions(u_test, predicted_states_A_full_rank)
mse_optimized, r2_A_noisy_full_rank = evaluate_predictions(u_test, predicted_states_A_noisy_full_rank)
mse_optimized, r2_A_triangular_matrix_pi = evaluate_predictions(u_test, predicted_states_A_triangular_matrix_pi)
mse_optimized, r2_A_circulant_matrix_pi = evaluate_predictions(u_test, predicted_states_A_circulant_matrix_pi)
mse_optimized, r2_A_diagonal_matrix_pi = evaluate_predictions(u_test, predicted_states_A_diagonal_matrix_pi)
mse_optimized, r2_A_symmetric_matrix_pi = evaluate_predictions(u_test, predicted_states_A_symmetric_matrix_pi)




# Output the R² results of the prediction
print("R² for A_k_symmetric:", r2_symmetric)
print("R² for A_k_l1:", r2_A_k_l1)
print("R² for A_k_grad:", r2_A_k_grad)
print("R² for A_k_l2:", r2_A_k_l2)
print("R² for A_k_low:", r2_A_k_low)
print("R² for A:", r2_A)
print("R² for A_noisy:", r2_A_noisy)
print("R² for A_full_rank:", r2_A_full_rank)
print("R² for A_noisy_full_rank:", r2_A_noisy_full_rank)
print("R² for A_odmd:", r2_odmd)
print("R² for A_circulant_matrix_pi:", r2_A_circulant_matrix_pi )
print("R² for A_triangular_matrix_pi:", r2_A_triangular_matrix_pi)
print("R² for A_symmetric_matrix_pi:", r2_A_symmetric_matrix_pi )
print("R² for A_diagonal_matrix_pi:", r2_A_diagonal_matrix_pi )


# Plot the matrix and eigenvalues
visualize_matrix(A, "A")
visualize_eigenvalues(A)
visualize_matrix(A_full_rank, "A")
visualize_eigenvalues(A_full_rank)
visualize_matrix(A_k_l1, 'A_l1')
visualize_eigenvalues(A_k_l1)
visualize_matrix(A_k_l2, "A_k_l2")
visualize_eigenvalues(A_k_l2)
visualize_matrix(A_k_low, "A_k_low")
visualize_eigenvalues(A_k_low)
visualize_matrix(A_grad, "A_grad")
visualize_eigenvalues(A_grad)
visualize_matrix(A_noisy, "A_noisy")
visualize_eigenvalues(A_noisy)
visualize_matrix(A_k_symmetric, "A_k_symmetric")
visualize_eigenvalues(A_k_symmetric)
visualize_matrix(online_dmd, "online_dmd")
visualize_eigenvalues(online_dmd)
visualize_matrix(A_noisy_full_rank, "A_noisy_full_rank")
visualize_eigenvalues(A_noisy_full_rank)
visualize_matrix(circulant_matrix_pi.A, "circulant_matrix_pi.A")
visualize_eigenvalues(circulant_matrix_pi.A)
visualize_matrix(triangular_matrix_pi.A, "triangular_matrix_pi.A")
visualize_eigenvalues(triangular_matrix_pi.A)
visualize_matrix(symmetric_matrix_pi.A, "symmetric_matrix_pi.A")
visualize_eigenvalues(symmetric_matrix_pi.A)
visualize_matrix(diagonal_matrix_pi.A, "diagonal_matrix_pi.A")
visualize_eigenvalues(diagonal_matrix_pi.A)
visualize_matrix(diagonal_matrix_pi.A, "unitary_matrix_pi.A")

































