# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pydmd import PiDMD


def visualize_eigenvalues(A, save_path=None):
    """
    Visualizes the eigenvalues of a square matrix A in the complex plane.
    This function plots the eigenvalues as points in the complex plane, along with the unit circle for reference.

    Parameters:
        A (ndarray): The input square matrix whose eigenvalues are to be visualized.
        save_path (str, optional): If provided, saves the plot as a PDF file to the specified path.

    Returns:
        None
    """
    # Compute the eigenvalues of the matrix A
    eigs, _ = np.linalg.eig(A)
    
    # Create a figure with fixed size for the plot
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    
    # Plot the axes and the unit circle for reference
    ax.axvline(x=0, color="k", lw=1)  # Vertical axis (Real axis)
    ax.axhline(y=0, color="k", lw=1)  # Horizontal axis (Imaginary axis)
    
    # Define the unit circle for visualization
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(t), np.sin(t), color="gray", linestyle="--")  # Unit circle

    # Plot the eigenvalues as blue points
    ax.scatter(eigs.real, eigs.imag, color="blue", alpha=0.7)  # Eigenvalues as points in the complex plane

    # Set axis labels
    ax.set_xlabel("Real", fontsize=12)
    ax.set_ylabel("Imag", fontsize=12)

    # Adjust the limits of the plot to ensure the unit circle and eigenvalues fit well
    max_val = max(1.0, np.max(np.abs(eigs.real)), np.max(np.abs(eigs.imag))) + 0.5
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    # Ensure the aspect ratio is equal to avoid distortion of the plot
    ax.set_aspect('equal', adjustable='box')

    # Adjust layout for a tight fit
    fig.tight_layout(pad=1.0)
    
    # Save the plot as a PDF if save_path is provided
    if save_path:
        plt.savefig(f"{save_path}", format='pdf', bbox_inches='tight')
    
    plt.show()


def gradient_descent_update(A_k, x_k, y_k, learning_rate):
    """
    Perform a gradient descent update step for the circulant matrix A_k using a single column x_k and corresponding y_k.
    
    Parameters:
    - A_k: Current matrix to be updated.
    - x_k: Input vector used in the update step.
    - y_k: Target vector used in the update step.
    - learning_rate: The learning rate for gradient descent.
    
    Returns:
    - A_k: Updated circulant matrix after the gradient descent step.
    """
    # Calculate the gradient
    grad = -2 * (y_k - A_k.dot(x_k)) * x_k.T
    # Update A_k
    A_k -= learning_rate * grad
    return A_k


def evaluate_predictions(actual, predicted):
    """
    Evaluate the prediction quality of a model using MSE and R^2 metrics.
    Adjusts for complex data by taking the real part.

    Parameters:
    - actual (np.ndarray): The actual data, which can be complex.
    - predicted (np.ndarray): The predicted data, which can be complex.

    Returns:
    - mse (float): Mean Squared Error of the predictions.
    - r2 (float): Coefficient of Determination (R^2) of the predictions.
    """
    actual_real = np.real(actual)
    predicted_real = np.real(predicted)

    mse = mean_squared_error(actual_real, predicted_real)
    r2 = r2_score(actual_real, predicted_real)
    
    return mse, r2

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

def run_online_gradient_descent(combined_matrix_noise, learning_rate, iterations):
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
        A_k = gradient_descent_update(A_k, x_k, y_k, learning_rate)  # Update A_k
        y_k_pred = A_k @ x_k
        # print(y_k_pred-y_k)
        y_estimates[:, idx] = y_k_pred.flatten()

    return A_k, y_estimates
 
def run_online_gradient_descent_tridiagonal(combined_matrix_noise, learning_rate, iterations):
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
        A_k = gradient_descent_update_tridiagonal(A_k, x_k, y_k, learning_rate)  # Update A_k
        y_k_pred = A_k @ x_k
        # print(y_k_pred-y_k)
        y_estimates[:, idx] = y_k_pred.flatten()

    return A_k, y_estimates
 
    

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



def visualize_matrix(A, title, save_path=None):
    vmax = np.abs(A).max()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    X, Y = np.meshgrid(np.arange(A.shape[1]+1), np.arange(A.shape[0]+1))
    ax.invert_yaxis()
    pos = ax.pcolormesh(X, Y, A.real, cmap="seismic", vmax=vmax, vmin=-vmax)
    # plt.title(f"{title}")

    cbar = fig.colorbar(pos, ax=ax)
    cbar.ax.tick_params(labelsize=10)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # 仅在提供了保存路径时保存图像
    if save_path:
        plt.savefig(f"{save_path}", format='pdf', bbox_inches='tight')
    
    # 显示图像
    plt.show()


def projection_onto_tridiagonal(A):
    """
    Efficiently project a matrix onto the space of tridiagonal matrices.
    """
    # Use NumPy's triu (upper triangle) and tril (lower triangle) functions
    # to keep only the main diagonal and the first diagonals above and below it.
    return np.triu(np.tril(A, 1), -1)

def predict_dmd(A, x_0, k):
    """
    Predict the future states from the 1st step to the k-th step using the Dynamic Mode Decomposition (DMD) method.

    Parameters:
        A (numpy.ndarray): The state transition matrix.
        x_0 (numpy.ndarray): The current state vector.
        k (int): The number of steps to predict.

    Returns:
        numpy.ndarray: A matrix containing the predicted states from the 1st step to the k-th step, 
                        with each column representing the state at a specific time step.
    """
    # Perform eigenvalue decomposition to get the eigenvalues and eigenvectors of A
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Project the initial state onto the eigenvector space, solving for the coefficients (b) in the eigenvector basis
    b = np.linalg.solve(eigenvectors, x_0).reshape(-1, 1)
    
    # Compute the eigenvalues raised to the powers from 1 to k (for time steps 1 to k)
    eigenvalue_powers = np.array([eigenvalues ** step for step in range(1, k + 1)]).T
    
    # Use broadcasting to compute the predicted states for each time step
    predicted_states = (eigenvectors @ (eigenvalue_powers * b)).real
    
    return predicted_states



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



def run_online_dmd_update(combined_matrix_noise, iterations):
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

    for idx in range(iterations):
        # Extract x_k and flatten it
        x_k = combined_matrix_noise[:, idx:idx+1].flatten()
        # Extract y_k and flatten it
        y_k = combined_matrix_noise[:, idx+1:idx+2].flatten()

        # Update DMD matrix using flattened x_k and y_k
        A_k = odmd.update(x_k, y_k)

    return odmd.A

def l1_soft_thresholding(A, lambda_):
    """
    Applies the soft-thresholding operator element-wise to the matrix A.
    Soft thresholding is used for L1 regularization (Lasso) during the optimization process.
    It shrinks the values of A toward zero by a factor determined by lambda_.

    Parameters:
        A (ndarray): The input matrix to which soft-thresholding will be applied.
        lambda_ (float): The regularization parameter that controls the amount of shrinkage.

    Returns:
        ndarray: The matrix after soft-thresholding has been applied element-wise.
    """
    return np.sign(A) * np.maximum(np.abs(A) - lambda_, 0.0)


def gradient_descent_update_l1(A_k, X_k, Y_k, lambda_reg, learning_rate):
    """
    Performs a single step of online gradient descent with L1 regularization.
    This involves computing the gradient of the loss function with respect to A_k,
    applying L1 regularization, and updating A_k by moving in the direction of the negative gradient
    scaled by the learning rate. It also includes a proximal step for L1 regularization (soft-thresholding).

    Parameters:
        A_k (ndarray): The current estimate of the matrix at step k.
        X_k (ndarray): The input data matrix at step k.
        Y_k (ndarray): The observation matrix at step k.
        lambda_reg (float): The regularization parameter for L1 regularization.
        learning_rate (float): The step size for the gradient descent.

    Returns:
        ndarray: The updated matrix A_k after applying the gradient descent update and L1 regularization.
    """
    # Compute the gradient of the loss function with respect to A_k
    grad = -2 * (Y_k - A_k @ X_k) @ X_k.T 
    # Update A_k in the direction of the negative gradient, scaled by the learning rate
    A_k -= learning_rate * grad
    # Apply soft-thresholding for L1 regularization
    A_k = l1_soft_thresholding(A_k, lambda_reg)
    return A_k


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


def gradient_descent_update_tridiagonal(A_k, X_k_new, Y_k_new, learning_rate):
    """
    Perform a single step of gradient descent and apply the tridiagonal matrix constraint.

    Parameters:
        A_k (numpy.ndarray): The current matrix to be updated.
        X_k_new (numpy.ndarray): The new input matrix (feature matrix).
        Y_k_new (numpy.ndarray): The new output matrix (target matrix).
        learning_rate (float): The learning rate for the gradient descent update.

    Returns:
        numpy.ndarray: The updated matrix A_k with the tridiagonal matrix constraint applied.
    """
    # Compute the gradient of the loss function with respect to A_k
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T
    
    # Update A_k by subtracting the gradient scaled by the learning rate
    A_k -= learning_rate * grad
    
    # Apply the tridiagonal matrix constraint using the projection operator
    A_k = projection_onto_tridiagonal(A_k)
    
    return A_k


def projection_onto_symmetric(A):
    """
    Project the matrix onto the space of symmetric matrices.
    
    This function takes a matrix A and returns the closest symmetric matrix
    by averaging A and its transpose.
    
    Parameters:
        A (numpy.ndarray): The matrix to be projected.
    
    Returns:
        numpy.ndarray: The symmetric matrix obtained by projecting A.
    """
    return (A + A.T) / 2


def gradient_descent_update_symmetric(A_k, X_k_new, Y_k_new, learning_rate):
    """
    Perform a single step of gradient descent with a symmetric matrix constraint.
    
    This function computes the gradient of the loss function with respect to A_k,
    updates A_k using the negative gradient scaled by the learning rate, and then
    projects the updated matrix onto the space of symmetric matrices.
    
    Parameters:
        A_k (numpy.ndarray): The current matrix to be updated.
        X_k_new (numpy.ndarray): The new input matrix (feature matrix).
        Y_k_new (numpy.ndarray): The new output matrix (target matrix).
        learning_rate (float): The learning rate for the gradient descent update.
    
    Returns:
        numpy.ndarray: The updated matrix A_k after the gradient descent update and projection.
    """
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T 
    A_k -= learning_rate * grad
    A_k = projection_onto_symmetric(A_k)  # Apply symmetric matrix constraint
    return A_k


def proximal_unclear_norm(A, lambda_reg):
    """
    Project the matrix onto the space constrained by a specified operator norm.
    
    This function applies a soft-thresholding operation to the singular values of A
    and projects the matrix onto the space of matrices with operator norm constraint.
    
    Parameters:
        A (numpy.ndarray): The matrix to be projected.
        lambda_reg (float): The regularization parameter that determines the threshold.
    
    Returns:
        numpy.ndarray: The matrix after applying the operator norm constraint.
    """
    # Perform SVD decomposition of matrix A
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # Apply operator norm constraint by soft-thresholding the singular values
    s = np.maximum(s - lambda_reg, 0)
    # Reconstruct the matrix using the modified singular values
    return U @ np.diag(s) @ Vt


def gradient_descent_update_unclear_norm(A_k, X_k_new, Y_k_new, lambda_reg, learning_rate):
    """
    Perform a single step of gradient descent with an operator norm constraint.
    
    This function computes the gradient of the loss function with respect to A_k,
    updates A_k using the negative gradient scaled by the learning rate, and then
    projects the updated matrix onto the space constrained by the operator norm.
    
    Parameters:
        A_k (numpy.ndarray): The current matrix to be updated.
        X_k_new (numpy.ndarray): The new input matrix (feature matrix).
        Y_k_new (numpy.ndarray): The new output matrix (target matrix).
        lambda_reg (float): The regularization parameter for the operator norm.
        learning_rate (float): The learning rate for the gradient descent update.
    
    Returns:
        numpy.ndarray: The updated matrix A_k after the gradient descent update and projection.
    """
    # Compute the gradient
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T
    # Perform gradient descent step
    A_k -= learning_rate * grad
    # Apply operator norm constraint
    A_k = proximal_unclear_norm(A_k, lambda_reg)
    return A_k


def gradient_descent_update_l2(A_k, X_k_col, Y_k_col, lambda_val, learning_rate):
    """
    Perform a single step of gradient descent with L2 regularization.
    
    This function computes the gradient of the loss function with respect to A_k,
    adds the L2 regularization term, and updates A_k using the negative gradient
    scaled by the learning rate.
    
    Parameters:
        A_k (numpy.ndarray): The current matrix to be updated.
        X_k_col (numpy.ndarray): The input vector (feature column).
        Y_k_col (numpy.ndarray): The output vector (target column).
        lambda_val (float): The L2 regularization parameter.
        learning_rate (float): The learning rate for the gradient descent update.
    
    Returns:
        numpy.ndarray: The updated matrix A_k after the gradient descent update with L2 regularization.
    """
    # Compute the gradient with L2 regularization
    gradient = 2 * (A_k @ X_k_col - Y_k_col) @ X_k_col.T + 2 * lambda_val * A_k

    # Update A_k using gradient descent
    A_k -= learning_rate * gradient
    return A_k






def run_online_gradient_descent_low_rank(combined_matrix_noise, lambda_reg,learning_rate, iterations):
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


    # Use the number of columns in combined_matrix_noise as the number of iterations
    for idx in range(iterations):
        x_k = combined_matrix_noise[:, idx:idx+1]  # Extract one column as x_k
        y_k = combined_matrix_noise[:, idx+1:idx+2]  # Extract the next column as y_k
        A_k = gradient_descent_update_unclear_norm(A_k, x_k, y_k, lambda_reg, learning_rate)  # Update A_k
    return A_k



def run_online_gradient_descent_l2(combined_matrix_noise, lambda_reg,learning_rate,  iterations):
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

    # Use the number of columns in combined_matrix_noise as the number of iterations
    for idx in range(iterations):
        x_k = combined_matrix_noise[:, idx:idx+1]  # Extract one column as x_k
        y_k = combined_matrix_noise[:, idx+1:idx+2]  # Extract the next column as y_k
        A_k = gradient_descent_update_l2(A_k, x_k, y_k, lambda_reg, learning_rate)  # Update A_k
       


    return A_k


def run_online_gradient_descent_symmetric(combined_matrix_noise, learning_rate, iterations):
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


    # Use the number of columns in combined_matrix_noise as the number of iterations
    for idx in range(iterations):
        x_k = combined_matrix_noise[:, idx:idx+1]  # Extract one column as x_k
        y_k = combined_matrix_noise[:, idx+1:idx+2]  # Extract the next column as y_k
        A_k = gradient_descent_update_symmetric(A_k, x_k, y_k, learning_rate)  # Update A_k

    return A_k

def plot_coordinate_comparison(time, X_actual, X_predicted_odmd, 
                               X_predicted_phy, X_predicted_pidmd, 
                               coordinate_labels=None, iterations=None, save_path=None):
    """
    Visualize the true, optimized prediction, online DMD prediction, and other variants for each coordinate.

    Parameters:
    - time: Time steps.
    - X_actual: numpy.ndarray, actual data matrix with shape (n, t).
    - X_predicted_odmd: numpy.ndarray, predicted data matrix using online DMD with shape (n, t).
    - X_predicted_phy: numpy.ndarray, predicted data matrix using physics-based DMD model with shape (n, t).
    - X_predicted_pidmd: numpy.ndarray, predicted data matrix using PIDMD model with shape (n, t).
    - coordinate_labels: list of str, optional, labels for each coordinate.
    - iterations: Number of time steps to display.
    - save_path: str, optional, directory path to save individual plots as files.
    """
    sns.set(style="white", context="talk")
    num_coordinates = X_actual.shape[0]

    for i in range(num_coordinates):
        plt.figure(figsize=(12, 8), dpi=300)

        # Set global font to Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 16

        # Plot actual data with a bold blue color
        plt.plot(time[:iterations], X_actual[i, :iterations], label='True data', linestyle='-', color='blue', linewidth=2.5)

        # Plot predicted data from different models with high-contrast colors
        plt.plot(time[:iterations], X_predicted_phy[i, :iterations], label='OPIDMD', linestyle='--', color='orange', linewidth=2.5)
        plt.plot(time[:iterations], X_predicted_odmd[i, :iterations], label='Online DMD', linestyle=':', color='green', linewidth=2.5)
        plt.plot(time[:iterations], X_predicted_pidmd[i, :iterations], label='piDMD', linestyle=(0, (5, 1)), color='purple', linewidth=2.5)

        # Set labels and title
        plt.xlabel('Time Step', fontsize=25)
        plt.ylabel(coordinate_labels[i] if coordinate_labels else f"Coordinate {i+1}", fontsize=25)

        # Legend settings
        legend = plt.legend(fontsize='x-large', handlelength=2, edgecolor='black', loc='upper right', frameon=True, fancybox=False)
        legend.get_frame().set_linewidth(1.5)

        # Tick settings
        plt.tick_params(axis='both', which='major', labelsize=20)

        # Grid settings
        plt.grid(True, linestyle='--', linewidth=0.5)

        # Tight layout
        plt.tight_layout()

        # Save image if save_path is provided
        if save_path:
            filename = f"{save_path}/comparison_{coordinate_labels[i] if coordinate_labels else 'Coordinate'}_{i+1}.pdf"
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            print(f"Image saved as {filename}")

        plt.show()
        
# Load the data
load_path = 'D:\\博士课题\\小论文\\environment load model\\论文代码\\Lorenz_data.npy'
save_path = "D:\\博士课题\\小论文\\environment load model\\论文代码\\A_k_optimized_Lorenz_grad.npy"
X = np.load(load_path)

# Set random seed for reproducibility
np.random.seed(0)

# Learning rates for different optimization constraints
learning_rate_tridiagonal = 0.000339  # Learning rate for tridiagonal matrix (noise rate 0.25)
learning_rate_grad = 0.000049  # Learning rate for gradient descent
learning_rate_l1 = 0.00006  # Learning rate for L1 regularization
learning_rate_symmetric = 0.000004  # Learning rate for symmetric matrix constraint
learning_rate_l2 = 0.00006  # Learning rate for L2 regularization
learning_rate_low_rank = 0.00006  # Learning rate for low-rank constraint

# Add non-stationary Gaussian noise to the data
noise_ratio = 0.25
X_noisy = add_nonstationary_gaussian_noise(X, noise_ratio)

# Regularization parameters
lambda_reg_l1 = 0.0000001  # Regularization parameter for L1 regularization
lambda_reg_l2 = 0.0001  # Regularization parameter for L2 regularization
lambda_reg_low_rank = 0.0000001  # Regularization parameter for low-rank constraint

# Set prediction steps
k_steps = 150  # Number of prediction steps
x_0 = X[:, -1 - k_steps]  # Use the (k+1)-th last time point as the initial state

# Split data into training and testing sets
X_train = X_noisy[:, :-1 - k_steps]  # Training data
X_test = X[:, -k_steps:]  # True data for validating the predictions

# Run OPIDMD to optimize the A matrix
iterations = X_train.shape[1]-1
A_k_grad, X_pred = run_online_gradient_descent(X_train, learning_rate_grad,  iterations)
A_k_symmetric = run_online_gradient_descent_symmetric(X_train , learning_rate_symmetric, iterations)
A_k_phy, X_pred = run_online_gradient_descent_tridiagonal(X_train, learning_rate_tridiagonal, iterations)
A_k_l1, X_pred = run_online_gradient_descent_l1(X_train,lambda_reg_l1, learning_rate_l1 , iterations)
online_dmd = run_online_dmd_update(X_train, iterations)
triangular_matrix_pi = PiDMD(manifold="diagonal", manifold_opt=2, compute_A=True).fit(X_train)
A_l2= run_online_gradient_descent_l2(X_train, lambda_reg_l2,learning_rate_l2,iterations)
A_k_low = run_online_gradient_descent_low_rank(X_train, lambda_reg_low_rank ,learning_rate_low_rank, iterations)


#pidmd
pidmd = triangular_matrix_pi.A

# # DMD
A_dmd = compute_dmd_matrix(X_train, 3)

# Perform k_steps step prediction
predicted_states_grad = predict_dmd(A_k_grad, x_0, k_steps)
predicted_states_phy = predict_dmd(A_k_phy, x_0, k_steps)
predicted_states_l1= predict_dmd(A_k_l1, x_0, k_steps)
predicted_states_symmetric= predict_dmd(A_k_symmetric, x_0, k_steps)
predicted_states_dmd = predict_dmd(A_dmd, x_0, k_steps)
predicted_states_odmd = predict_dmd(online_dmd, x_0, k_steps)
predicted_states_pidmd = predict_dmd(pidmd, x_0, k_steps)
predicted_states_pidmd = predict_dmd(pidmd, x_0, k_steps)
predicted_states_l2 = predict_dmd(A_l2, x_0, k_steps)
predicted_states_low = predict_dmd(A_k_low, x_0, k_steps)

# Compute the prediction error
mse_optimized, r2_grad = evaluate_predictions(X_test, predicted_states_grad)
mse_optimized_phy, r2_phy = evaluate_predictions(X_test, predicted_states_phy)
mse_optimized_phy, r2_l1 = evaluate_predictions(X_test, predicted_states_l1)
mse_optimized_phy, r2_symmetric = evaluate_predictions(X_test, predicted_states_symmetric)
mse_dmd, r2_dmd = evaluate_predictions(X_test, predicted_states_dmd)
mse_dmd, r2_odmd = evaluate_predictions(X_test, predicted_states_odmd)
mse_dmd, r2_pidmd = evaluate_predictions(X_test, predicted_states_pidmd)
mse_dmd, r2_l2 = evaluate_predictions(X_test, predicted_states_l2)
mse_dmd, r2_low = evaluate_predictions(X_test, predicted_states_low)

# Output the error results
print("R² for A_k_grad:", r2_grad)
print("R² for A_k_phy:", r2_phy)
print("R² for A_k_l1:", r2_l1)
print("R² for A_k_symmetric:", r2_symmetric)
print("R² for A_dmd:", r2_dmd)
print("R² for A_odmd:", r2_odmd)
print("R² for A_pidmd:", r2_pidmd)
print("R² for A_l2:", r2_l2)
print("R² for A_low:", r2_low)

# Visualize the A matrix
visualize_matrix(A_k_grad, "A_k_grad")
visualize_eigenvalues(A_k_grad)
visualize_matrix(A_k_phy, "A_k_phy")
visualize_eigenvalues(A_k_phy)
visualize_matrix(online_dmd, "A_odmd")
visualize_eigenvalues(online_dmd)
visualize_matrix(A_dmd, "A_dmd")
visualize_eigenvalues(A_dmd)
visualize_matrix(pidmd, "pidmd")
visualize_eigenvalues(pidmd)


time = np.linspace(0, 150, 150) 


# Usage example with dummy variables for `predict_dmd` results
plot_coordinate_comparison(time, X_test, predicted_states_odmd,
                            predicted_states_phy, predicted_states_pidmd,
                            coordinate_labels=['x', 'y', 'z'], iterations=150)























