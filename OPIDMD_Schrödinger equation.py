# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.sparse.linalg import spsolve
from visualization_functions import visualize_matrix,compute_and_plot_singular_values_scatter, plot_frobenius_diff,visualize_eigenvalues
np.random.seed(0)
from sklearn.metrics import mean_squared_error, r2_score
from pydmd import PiDMD
from matplotlib.ticker import MaxNLocator
from scipy.sparse import diags
# Adjusted Newmark-beta method function
from scipy.linalg import circulant  # 确保导入 circulant 函数
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

    This function calculates the symmetrized version of a given matrix
    by averaging the matrix and its transpose.

    Parameters:
    - A (numpy.ndarray): The input matrix to be projected.

    Returns:
    - numpy.ndarray: The symmetric matrix obtained by the projection.
    """
    return (A + A.T) / 2


def gradient_descent_update_symmetric(A_k, X_k_new, Y_k_new, learning_rate):
    """
    Performs a single update step using online gradient descent with a symmetric matrix constraint.

    This function computes the gradient of the loss function with respect to A_k,
    updates A_k in the direction of the negative gradient scaled by the learning rate,
    and enforces the symmetric matrix constraint by projecting A_k onto the space of symmetric matrices.

    Parameters:
    - A_k (numpy.ndarray): The current estimate of the matrix to be updated.
    - X_k_new (numpy.ndarray): The new input data (features) for the current step.
    - Y_k_new (numpy.ndarray): The corresponding output data (targets) for the current step.
    - learning_rate (float): The learning rate for gradient descent.

    Returns:
    - numpy.ndarray: The updated symmetric matrix A_k.
    """
    # Compute the gradient of the loss function
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T
    # Update A_k in the direction of the negative gradient
    A_k -= learning_rate * grad
    # Enforce the symmetric matrix constraint
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

def run_online_gradient_descent_circulant(combined_matrix_noise, learning_rate, D, iterations):
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
        A_k = gradient_descent_update_circulant(A_k, x_k, y_k, learning_rate)  # Update A_k
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


def setup_simulation(Lx, Nx, dt, T):
    ''' 
    Set up initial parameters and grids.
    
    This function calculates the spatial step size (dx), the number of time steps (Nt),
    and generates a spatial grid (x) for the given domain length (Lx) and number of points (Nx).
    
    Parameters:
    Lx - Length of the spatial domain
    Nx - Number of spatial grid points
    dt - Time step size
    T  - Total time for the simulation
    
    Returns:
    dx - Spatial step size
    Nt - Number of time steps
    x  - Spatial grid points (array)
    '''
    dx = Lx / Nx  # Spatial step size
    Nt = round(T / dt)  # Number of time steps
    x = np.linspace(-Lx / 2, Lx / 2, Nx)  # Spatial grid from -Lx/2 to Lx/2
    
    return dx, Nt, x

# def initialize_wave_packet(x, x0=0, sigma=0.5, k0=2):
def initialize_wave_packet(x, x0=0, sigma=1, k0=20):
    ''' 
    Initialize wave packet (initial conditions).
    
    This function initializes the wave packet as a Gaussian wave packet with an initial momentum.
    
    Parameters:
    x     - Spatial grid points (array)
    x0    - Initial position of the wave packet (default = 0)
    sigma - Width of the wave packet (default = 0.5)
    k0    - Initial momentum of the wave packet (default = 2)
    
    Returns:
    psi0  - Initial wave function (complex array)
    '''
    psi0 = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)  # Initial wave function
    return psi0

def compute_laplacian_operator(Nx, dx):
    ''' 
    Compute Laplacian operator (with periodic boundary conditions).
    
    This function discretizes the Laplacian operator using finite differences and applies
    periodic boundary conditions at the edges of the spatial grid.
    
    Parameters:
    Nx - Number of spatial grid points
    dx - Spatial step size
    
    Returns:
    L_operator - Discrete Laplacian operator (matrix)
    '''
    diagonals = [-2 * np.ones(Nx), np.ones(Nx - 1), np.ones(Nx - 1)]  # Main diagonal and off-diagonals
    L_operator = diags(diagonals, [0, 1, -1], shape=(Nx, Nx)).toarray() / dx**2  # Laplacian matrix
    
    # Apply periodic boundary conditions: connecting first and last grid points
    L_operator[0, -1] = 1 / dx**2
    L_operator[-1, 0] = 1 / dx**2
    
    return L_operator

def setup_crank_nicolson(hbar, m, dt, L_operator):
    ''' 
    Set up Crank-Nicolson method matrices.
    
    This function sets up the A and B matrices required for the Crank-Nicolson time-stepping scheme,
    which is used for solving the Schrödinger equation numerically.
    
    Parameters:
    hbar        - Reduced Planck's constant (for simplicity set to 1)
    m           - Mass of the particle (for simplicity set to 1)
    dt          - Time step size
    L_operator  - Discrete Laplacian operator
    
    Returns:
    A_sparse    - Sparse matrix A for Crank-Nicolson scheme
    B_sparse    - Sparse matrix B for Crank-Nicolson scheme
    '''
    A = np.eye(len(L_operator)) + 1j * dt * (-hbar**2 / (2 * m)) * L_operator / 2  # Matrix A
    B = np.eye(len(L_operator)) - 1j * dt * (-hbar**2 / (2 * m)) * L_operator / 2  # Matrix B
    
    # Convert matrices to sparse format for computational efficiency
    A_sparse = diags([A.diagonal(), A.diagonal(1), A.diagonal(-1)], [0, 1, -1]).tocsc()
    B_sparse = diags([B.diagonal(), B.diagonal(1), B.diagonal(-1)], [0, 1, -1]).tocsc()
    
    return A_sparse, B_sparse

def solve_schrodinger(Nx, Nt, psi0, A_sparse, B_sparse):
    ''' 
    Solve Schrödinger equation using Crank-Nicolson method.
    
    This function evolves the wave function over time using the Crank-Nicolson method, storing
    the probability density |ψ(x, t)|^2 at each time step in the psi_store array.
    
    Parameters:
    Nx        - Number of spatial grid points
    Nt        - Number of time steps
    psi0      - Initial wave function (complex array)
    A_sparse  - Sparse matrix A for Crank-Nicolson scheme
    B_sparse  - Sparse matrix B for Crank-Nicolson scheme
    
    Returns:
    psi_store - Array storing the probability density |ψ(x, t)|^2 over time
    '''
    psi = psi0.copy()  # Initialize wave function
    psi_store = np.zeros((Nx, Nt))  # Array to store |ψ(x, t)|^2
    
    # Time-stepping loop using Crank-Nicolson method
    for n in range(Nt):
        psi_store[:, n] = np.abs(psi)**2  # Store probability density at current time step
        psi = spsolve(A_sparse, B_sparse @ psi)  # Solve for the next time step using Crank-Nicolson
    
    return psi_store

def plot_wave_function_surface(psi_store, x, T, Nt, dpi=300, save_path=None):
    ''' 
    Generate a 3D surface plot for wave function evolution.

    This function creates a 3D surface plot of the probability density 
    |ψ(x, t)|² over time and space, visualizing the evolution of the wave function.

    Parameters:
    - psi_store (numpy.ndarray): Array storing the probability density |ψ(x, t)|² at each time step.
    - x (numpy.ndarray): Spatial grid points.
    - T (float): Total simulation time.
    - Nt (int): Number of time steps in the simulation.
    - dpi (int): Resolution of the figure in dots per inch (default: 300).
    - save_path (str or None): Path to save the plot as a PDF. If None, the plot is displayed but not saved.

    Returns:
    None
    '''
    # Set the font style for the plot
    plt.rcParams['font.family'] = 'Times New Roman'
    fig = plt.figure(dpi=dpi)  # Create a figure with specified DPI for high resolution
    T_mesh, X_mesh = np.meshgrid(np.linspace(0, T, Nt), x)  # Create a meshgrid for time and space
    ax = fig.add_subplot(111, projection='3d')  # Add a 3D subplot to the figure

    # Plot the surface of the wave function's probability density
    surface = ax.plot_surface(T_mesh, X_mesh, psi_store, cmap='viridis', edgecolor='none')

    # Set axis labels
    ax.set_xlabel('Time t', fontsize=10)
    ax.set_ylabel('Space x', fontsize=10)

    # Optionally set the z-axis label and title (commented out in this version)
    # ax.set_zlabel(r'$|\psi(x, t)|^2$', fontsize=10)
    # ax.set_title('Probability Density Evolution of Wave Function', fontsize=12)

    # Optionally add a color bar to the plot (commented out in this version)
    # fig.colorbar(surface, ax=ax)

    # Save the figure to the specified path if provided
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Figure saved as {save_path}")

    # Display the plot
    plt.show()

    
    

def plot_wave_function_heatmap(psi_store, x, T, Lx, dpi=300):
    ''' 
    Plot 2D heatmap for wave function evolution.
    
    This function generates a 2D heatmap of the wave function's probability density evolution
    over time and space.
    
    Parameters:
    psi_store - Array storing the probability density |ψ(x, t)|^2 over time
    x         - Spatial grid points (array)
    T         - Total time for the simulation
    Lx        - Length of the spatial domain
    dpi       - Dots per inch (DPI) for figure resolution (default = 300)
    '''
    plt.figure(dpi=dpi)  # Set figure DPI for high resolution
    plt.imshow(psi_store, aspect='auto', extent=[0, T, -Lx / 2, Lx / 2], origin='lower', cmap='plasma')
    plt.colorbar(label=r'$|\psi(x, t)|^2$')  # Color bar for the probability density
    plt.xlabel('Time t')
    plt.ylabel('Space x')
    plt.title('Probability Density Evolution of Wave Function (2D Heatmap)')
    plt.show()

def l1_soft_thresholding(A, lambda_):
    """
    Applies soft thresholding to each element of matrix A.

    Soft thresholding is commonly used for L1 regularization (Lasso) in optimization. 
    It shrinks the values of A toward zero by a threshold defined by lambda_.

    Parameters:
    - A (numpy.ndarray): The input matrix to apply soft thresholding.
    - lambda_ (float): The threshold parameter that determines the degree of shrinkage.

    Returns:
    - numpy.ndarray: The matrix after applying soft thresholding, with elements shrunk toward zero.
    """
    return np.sign(A) * np.maximum(np.abs(A) - lambda_, 0.0)

def gradient_descent_update_l1(A_k, X_k, Y_k, lambda_reg, learning_rate):
    """
    Performs a single update step using online gradient descent with L1 regularization.

    This function computes the gradient of the loss function with respect to A_k,
    updates A_k in the direction of the negative gradient scaled by the learning rate,
    and applies L1 regularization via the proximal operator (soft thresholding).

    Parameters:
    - A_k (numpy.ndarray): The current estimate of the matrix to be updated.
    - X_k (numpy.ndarray): The input data (features) for the current step.
    - Y_k (numpy.ndarray): The corresponding output data (targets) for the current step.
    - lambda_reg (float): The regularization parameter for L1 regularization.
    - learning_rate (float): The learning rate for gradient descent.

    Returns:
    - numpy.ndarray: The updated matrix A_k after applying gradient descent and L1 regularization.
    """
    # Compute the gradient of the loss function
    grad = -2 * (Y_k - A_k @ X_k) @ X_k.T 
    # Update A_k in the direction of the negative gradient
    A_k -= learning_rate * grad
    # Apply soft thresholding for L1 regularization
    A_k = l1_soft_thresholding(A_k, lambda_reg)
    return A_k


def plot_solution(x, t, u):
    """
    Plots the solution of the 1D convection-diffusion equation with a scientific aesthetic.

    Parameters:
    x (array): Spatial grid.
    t (array): Time grid.
    u (2D array): Solution matrix with shape (Nx, Nt).
    """
    
    # Create a meshgrid for the space and time variables
    X, T = np.meshgrid(t, x)
    
    # Create the figure and axes with higher DPI for better resolution
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface with high detail and a more scientific colormap
    surf = ax.plot_surface(T, X, u, cmap='viridis', edgecolor='none', alpha=0.9)
    
    # Add a color bar with a scientific color scheme
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)
    cbar.set_label('u(x, t)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Customize axes labels with appropriate padding and larger font size
    ax.set_xlabel('Time', fontsize=14, labelpad=20)
    ax.set_ylabel('Space', fontsize=14, labelpad=20)
    ax.set_zlabel('u(x, t)', fontsize=14, labelpad=10)
    
    # Set the title and use a clean font
    ax.set_title('1D Convection-Diffusion Equation Solution', fontsize=16, pad=20)
    
    # Set the 3D plot view for better visualization
    ax.view_init(elev=30, azim=135)
    
    # Improve axis ticks and grid lines for a clean scientific look
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.zaxis.set_tick_params(labelsize=12)

    # Add a subtle grid for a better visual cue
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Tight layout to reduce white space
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def proximal_unclear_norm(A, lambda_reg):
    """
    Projects a matrix onto the space of matrices with a specified operator norm constraint.

    This function uses Singular Value Decomposition (SVD) to apply a proximal operator
    that enforces the operator norm constraint by shrinking singular values.

    Parameters:
    - A (numpy.ndarray): The input matrix to be projected.
    - lambda_reg (float): The regularization parameter for the operator norm constraint.

    Returns:
    - numpy.ndarray: The projected matrix with modified singular values.
    """
    # Perform Singular Value Decomposition (SVD) of the matrix A
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # Apply the operator norm constraint by shrinking the singular values
    s = np.maximum(s - lambda_reg, 0)
    # Reconstruct the matrix using the modified singular values
    return U @ np.diag(s) @ Vt



def project_to_circulant_matrix(M):
    """
    Projects a square matrix M to the nearest circulant matrix.

    Parameters:
    - M: A square matrix of shape (n, n).

    Returns:
    - C: The circulant matrix closest to M.
    """
    n = M.shape[0]
    # Create an index matrix to retrieve elements along circulant diagonals
    indices = (np.arange(n)[:, None] + np.arange(n)) % n  # Shape (n, n)
    # Retrieve elements along circulant diagonals
    M_shifted = M[np.arange(n)[:, None], indices]  # Shape (n, n)
    # Compute the average value for each circulant diagonal
    c = M_shifted.mean(axis=0)
    # Generate the circulant matrix
    C = circulant(c)
    return C

def gradient_descent_update_circulant(A_k, X_k_new, Y_k_new, learning_rate):
    """
    Performs a single update step using online gradient descent with a circulant matrix constraint.

    Parameters:
    - A_k (numpy.ndarray): The current estimate of the matrix.
    - X_k_new (numpy.ndarray): The new input data (features).
    - Y_k_new (numpy.ndarray): The corresponding output data (targets).
    - learning_rate (float): The learning rate for gradient descent.

    Returns:
    - numpy.ndarray: The updated circulant matrix A_k.
    """
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T
    A_k -= learning_rate * grad
    return project_to_circulant_matrix(A_k)

def gradient_descent_update_unclear_norm(A_k, X_k_new, Y_k_new, lambda_reg, learning_rate):
    """
    Performs a single update step using online gradient descent with an operator norm constraint.

    Parameters:
    - A_k (numpy.ndarray): The current estimate of the matrix.
    - X_k_new (numpy.ndarray): The new input data (features).
    - Y_k_new (numpy.ndarray): The corresponding output data (targets).
    - lambda_reg (float): The regularization parameter for the operator norm.
    - learning_rate (float): The learning rate for gradient descent.

    Returns:
    - numpy.ndarray: The updated matrix A_k after applying the gradient descent step and operator norm regularization.
    """
    # Compute the gradient
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T
    # Perform the gradient descent step
    A_k -= learning_rate * grad
    # Apply the operator norm constraint
    A_k = proximal_unclear_norm(A_k, lambda_reg)
    return A_k

def predict_dmd(A, x_0, k):
    """
    Predicts future states using the Dynamic Mode Decomposition (DMD) method from step 1 to step k.

    Parameters:
    - A (numpy.ndarray): The state transition matrix.
    - x_0 (numpy.ndarray): The current state vector.
    - k (int): The number of prediction steps.

    Returns:
    - numpy.ndarray: The predicted state matrix from step 1 to step k, where each column represents the state at a specific step.
    """
    # Perform eigenvalue decomposition to get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Project the initial state onto the eigenvector space and reshape as a column vector
    b = np.linalg.solve(eigenvectors, x_0).reshape(-1, 1)
    
    # Compute the powers of the eigenvalues for each step and transpose for broadcasting
    eigenvalue_powers = np.array([eigenvalues ** step for step in range(1, k + 1)]).T
    
    # Compute the predicted states using broadcasting
    predicted_states = (eigenvectors @ (eigenvalue_powers * b)).real
    
    return predicted_states

def gradient_descent_update_l2(A_k, X_k_col, Y_k_col, lambda_val, learning_rate):
    """
    Performs a single update step using gradient descent with L2 regularization.

    Parameters:
    - A_k (numpy.ndarray): The current estimate of the matrix.
    - X_k_col (numpy.ndarray): The input data (features) for the current step.
    - Y_k_col (numpy.ndarray): The corresponding output data (targets) for the current step.
    - lambda_val (float): The regularization parameter for L2 regularization.
    - learning_rate (float): The learning rate for gradient descent.

    Returns:
    - numpy.ndarray: The updated matrix A_k after the gradient descent step.
    """
    # Compute the gradient of the loss function with L2 regularization
    gradient = 2 * (A_k @ X_k_col - Y_k_col) @ X_k_col.T + 2 * lambda_val * A_k
    # Update A_k in the direction of the negative gradient
    A_k -= learning_rate * gradient
    return A_k



# Parameter settings
noise_ratio = 0.25    # Noise ratio
rank = 20             # Rank of the DMD matrix
learning_rate_grad = 0.001  # Learning rate for online gradient descent
learning_rate_symmetric = 0.01  # Learning rate for online gradient descent with symmetric matrix constraint
learning_rate_circu = 0.014  # Learning rate for online gradient descent with circulant matrix constraint
lambda_reg_l1 = 0.000000099  # Regularization parameter for L1 regularization
lambda_reg_low_rank = 0.00014  # Regularization parameter for low-rank constraint
lambda_reg_l2 = 0.037  # Regularization parameter for L2 regularization


# Parameter settings
Lx = 10            # Spatial domain length
Nx = 200           # Number of spatial grid points
dt = 0.0001        # Time step size (small for numerical stability)
T = 1              # Total time
hbar = 1           # Reduced Planck's constant (simplified to 1)
m = 1              # Mass of the particle (simplified to 1)
time = np.arange(0, 1, dt)
# Step 1: Set up simulation parameters
dx, Nt, x = setup_simulation(Lx, Nx, dt, T)

# Step 2: Initialize the wave packet
psi0 = initialize_wave_packet(x)

# Step 3: Compute Laplace operator
L_operator = compute_laplacian_operator(Nx, dx)

# Step 4: Set up Crank-Nicolson method matrices
A_sparse, B_sparse = setup_crank_nicolson(hbar, m, dt, L_operator)

# Step 5: Solve the Schrödinger equation
u = solve_schrodinger(Nx, Nt, psi0, A_sparse, B_sparse)

# Add non-stationary Gaussian noise
noisy_u = add_nonstationary_gaussian_noise(u, noise_ratio)

# Step 6: Plot the results with 300 dpi
plot_wave_function_surface(u, x, T, Nt, dpi=300)  # 3D surface plot of the original data
plot_wave_function_surface(noisy_u, x, T, Nt, dpi=300)  # 3D surface plot of the noisy data

# Prediction settings
k_steps = 150                  # Number of prediction steps
# k_steps = 50                 # Alternative number of prediction steps
u_0 = u[:, -1 - k_steps]       # Use the (k+1)-th last time point as the initial state
u_train = noisy_u[:, :-1 - k_steps]  # Training data
u_test = noisy_u[:, -k_steps:]       # True data for validating the predictions
iterations = u_train.shape[1] - 1    # Number of iterations

# Compute DMD matrices
A = compute_dmd_matrix(u[:, :-1 - k_steps], rank)  # DMD matrix for reduced data
A_full_rank = compute_dmd_matrix(u, None)         # Full-rank DMD matrix for original data
A_noisy = compute_dmd_matrix(noisy_u, rank)       # DMD matrix for noisy data (reduced rank)
A_noisy_full_rank = compute_dmd_matrix(noisy_u, None)  # Full-rank DMD matrix for noisy data

# Optimize DMD matrices: Using symmetric constraints and online DMD updates
A_k_symmetric, frobenius_diffs_opidmd = run_online_gradient_descent_symmetric(u_train, learning_rate_symmetric, A, iterations)
A_k_circulant, frobenius_diffs_opidmd = run_online_gradient_descent_circulant(u_train, learning_rate_circu, A, iterations)
A_k_l1, frobenius_diffs_opidmd = run_online_gradient_descent_l1(u_train, lambda_reg_l1, learning_rate_grad, iterations)
A_k_l2, frobenius_diffs_opidmd = run_online_gradient_descent_l2(u_train, lambda_reg_l2, learning_rate_grad, A, iterations)
A_grad, frobenius_diffs_opidmd = run_online_gradient_descent_l2(u_train, 0, learning_rate_grad, A, iterations)
A_k_low, frobenius_diffs_opidmd = run_online_gradient_descent_low_rank(u_train, lambda_reg_low_rank, learning_rate_grad, A, iterations)
online_dmd, frobenius_diffs_odmd = run_online_dmd_update(u_train, A, iterations)

# Compute matrices with specific structures using piDMD
triangular_matrix_pi = PiDMD(manifold="lowertriangular", compute_A=True).fit(u_train)
circulant_matrix_pi = PiDMD(manifold="circulant", compute_A=True).fit(u_train)
diagonal_matrix_pi = PiDMD(manifold="diagonal", compute_A=True).fit(u_train)
symmetric_matrix_pi = PiDMD(manifold="symmetric", compute_A=True).fit(u_train)


# Perform k-step prediction
predicted_states_symmetric = predict_dmd(A_k_symmetric, u_0, k_steps)
predicted_states_circulant = predict_dmd(A_k_circulant, u_0, k_steps)
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
mse_optimized, r2_circulant = evaluate_predictions(u_test, predicted_states_circulant)
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
print("R² for A_k_circulant:", r2_circulant)
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


# Plot the matrix visualization and eigenvalues
visualize_matrix(A, "A")
visualize_eigenvalues(A)
visualize_matrix(A_k_l1, "A_k_l1")
visualize_eigenvalues(A_k_l1)
visualize_matrix(A_k_circulant, "A_k_circulant")
visualize_eigenvalues(A_k_circulant )
visualize_matrix(A_k_l2, "A_k_l2")
visualize_eigenvalues(A_k_l2)
visualize_matrix(A_k_low, "A_k_low")
visualize_eigenvalues(A_k_low)
visualize_matrix(A_grad, "A_grad")
visualize_eigenvalues(A_grad)
visualize_matrix(A_noisy, "A_noisy")
visualize_eigenvalues(A_noisy)
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












































