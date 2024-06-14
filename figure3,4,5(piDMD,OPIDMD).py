# -*- coding: utf-8 -*-
import numpy as np
from visualization_functions import visualize_matrix, visualize_eigenvalues, plot_frobenius_diff, plot_gradient_descent_paths
from gradient_descent_functions import gradient_descent_update_triangular
from gradient_descent_functions import  gradient_descent_update_tridiagonal
from generate_data import  generate_stable_triangular_matrix , generate_sequential_control_data, generate_random_data
from gradient_descent_functions import  gradient_descent_update_symmetric
from scipy.linalg import circulant
from gradient_descent_functions import  gradient_descent_update_circulant
from pydmd import DMD
from pydmd import PiDMD
import matplotlib.pyplot as plt
np.random.seed(5)  # Set the random seed to 5


def visualize_matrices(matrices, titles, aspect='real', save_path="C:\\Users\\HIT\\Desktop"):
    num_matrices = len(matrices)
    fig, axs = plt.subplots(1, num_matrices, figsize=(num_matrices * 5, 5))

    # Process the matrices based on the specified aspect
    processed_matrices = []
    if aspect == 'real':
        processed_matrices = [np.real(matrix) for matrix in matrices]
    elif aspect == 'imaginary':
        processed_matrices = [np.imag(matrix) for matrix in matrices]
    elif aspect == 'magnitude':
        processed_matrices = [np.abs(matrix) for matrix in matrices]
    else:
        raise ValueError("Aspect must be 'real', 'imaginary', or 'magnitude'")

    # Find global maximum for color range normalization across all processed matrices
    vmax = max([np.abs(matrix).max() for matrix in processed_matrices])
    vmin = -vmax

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.size'] = 22

    for i, (matrix, title) in enumerate(zip(processed_matrices, titles)):
        ax = axs[i]
        cax = ax.matshow(matrix, cmap="seismic", vmax=vmax, vmin=vmin)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, y=-0.15)
        ax.grid(visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(wspace=0.2, hspace=0)
    plt.savefig(f"{save_path}\\matrices_visualization.pdf", format='pdf', bbox_inches='tight')
    plt.show()



def generate_data_matrix(D, rows, cols):
    """
    Generate data matrix Y using the matrix multiplication Y = D * X.

    Parameters:
    - D (np.array): The coefficient matrix of shape (n, n).
    - rows (int): Number of rows for the matrix X (should be equal to n, the number of columns of D).
    - cols (int): Number of columns for the matrix X.

    Returns:
    - Y (np.array): The resultant data matrix of shape (n, cols).
    """
    # Generate random matrix X with dimensions matching the requirements for multiplication with D
    X = np.random.randn(rows, cols)
    
    # Compute the matrix Y as the product of D and X
    Y = np.dot(D, X)
    
    return X,Y


def run_online_gradient_descent_triangular(A_k, D, X_k, learning_rate, iterations):
    """
    Run online gradient descent to update matrix A_k using generated data.
    
    Parameters:
    - A_k: Initial matrix A_k to be updated.
    - D: The upper triangular matrix used to generate Y_k_new.
    - X_k: Initial state vector used to generate data.
    - learning_rate: The learning rate for gradient descent.
    - iterations: Number of iterations to run the gradient descent.
    
    Returns:
    - A_k: Updated matrix A_k after gradient descent.
    - frobenius_diffs: List of Frobenius norm differences between A_k and D over iterations.
    """
    n = A_k.shape[0]
    frobenius_diffs = []
    X_k = X_k_inital  # Initialize the state vector
    
    for _ in range(iterations):
        # Generate new random state vector X_k for this iteration
        X_k = np.random.randn(n, 1)
        
        # Generate new data points X_k_new and Y_k_new based on D and X_k
        X_k_new, Y_k_new = generate_random_data(D, X_k, n)
        
        # Update A_k using gradient descent with the newly generated data
        A_k = gradient_descent_update_triangular(A_k, X_k_new, Y_k_new, learning_rate)
        
        # Calculate the Frobenius norm difference between the updated A_k and D
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)
        
        # Update X_k for the next iteration
        # X_k = Y_k
    
    return A_k, frobenius_diffs




def run_experiments(D, X_k, num_experiments, n, learning_rate, iterations):
    """
    Run multiple experiments of online gradient descent to observe performance.
    
    Parameters:
    - D: The upper triangular matrix used to generate Y_k_new.
    - X_k: Initial state vector used to generate data.
    - num_experiments: Number of experiments to run.
    - n: Dimension of the matrices and vectors.
    - learning_rate: The learning rate for gradient descent.
    - iterations: Number of iterations to run the gradient descent in each experiment.
    
    Returns:
    - all_diffs: List of Frobenius norm differences for each experiment.
    """
    all_diffs = []
    
    # Run the specified number of experiments
    for exp in range(num_experiments):
        # Initialize a random A_k matrix for each experiment
        A_k = np.random.randn(n, n)
        
        # Run online gradient descent for the current experiment
        A_k_optimized, frobenius_diffs = run_online_gradient_descent_triangular(A_k, D, X_k, learning_rate, iterations)
        
        # Store the Frobenius norm differences for the current experiment
        all_diffs.append(frobenius_diffs)
    
    return all_diffs





def run_online_gradient_descent_Tri_Diagonal(A_k, D,X_k_inital, learning_rate, iterations):
    """
    Run online gradient descent to update matrix A_k using generated data.
    
    Parameters:
    - A_k: Initial matrix A_k to be updated.
    - D: The Tri_Diagonal matrix used to generate Y_k_new.
    - learning_rate: The learning rate for gradient descent.
    - iterations: Number of iterations to run the gradient descent.
    
    Returns:
    - A_k: Updated matrix A_k after gradient descent.
    - frobenius_diff: The Frobenius norm difference between A_k and D.
    """
    n = A_k.shape[0]
    frobenius_diffs = []
    X_k = X_k_inital
    for _ in range(iterations):
        X_k = np.random.randn(n, 1)
        X_k_new, Y_k_new = generate_random_data(D, X_k, n)
        A_k = gradient_descent_update_tridiagonal(A_k, X_k_new, Y_k_new , learning_rate)
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)
    return A_k, frobenius_diffs

def projection_onto_symmetric(A):
    """
    Project a given matrix onto the space of symmetric matrices.
    
    Parameters:
    - A: The matrix to be projected.
    
    Returns:
    - The symmetric matrix resulting from the projection.
    
    Notes:
    - The projection is achieved by averaging the matrix with its transpose.
    """
    return (A + A.T) / 2



def run_online_gradient_descent_symmetric(A_k, D, X_k_inital,learning_rate, iterations):
    """
    Run online gradient descent to update matrix A_k using generated data.
    
    Parameters:
    - A_k: Initial matrix A_k to be updated.
    - D: The symmetric matrix used to generate Y_k_new.
    - learning_rate: The learning rate for gradient descent.
    - iterations: Number of iterations to run the gradient descent.
    
    Returns:
    - A_k: Updated matrix A_k after gradient descent.
    - frobenius_diff: The Frobenius norm difference between A_k and D.
    """
    n = A_k.shape[0]
    frobenius_diffs = []
    X_k = X_k_inital
    for _ in range(iterations):
        X_k = np.random.randn(n, 1)
        X_k_new, Y_k_new = generate_random_data(D, X_k, n)
        A_k = gradient_descent_update_symmetric(A_k, X_k_new,Y_k_new ,  learning_rate)
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)
    return A_k, frobenius_diffs



def run_online_gradient_descent_circulant(A_k, D, X_k_inital, learning_rate, iterations):
    """
    Run online gradient descent to update matrix A_k using generated data.
    
    Parameters:
    - A_k: Initial matrix A_k to be updated.
    - D: The circulant matrix used to generate Y_k_new.
    - learning_rate: The learning rate for gradient descent.
    - iterations: Number of iterations to run the gradient descent.
    
    Returns:
    - A_k: Updated matrix A_k after gradient descent.
    - frobenius_diff: The Frobenius norm difference between A_k and D.
    """
    n = A_k.shape[0]
    frobenius_diffs = []
    X_k = X_k_inital
    for _ in range(iterations):
        X_k = np.random.randn(n, 1)
        X_k_new, Y_k_new = generate_random_data(D, X_k, n)
        A_k = gradient_descent_update_circulant(A_k, X_k_new, Y_k_new , learning_rate)
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)
    return A_k, frobenius_diffs


def gradient_descent_update(A_k, X_k_new, Y_k_new, learning_rate):
    """
    Perform a single step update using online gradient descent with circulant matrix constraint.
    
    Parameters:
    - A_k: Current matrix to be updated.
    - X_k_new: New data point (state vector) used for the update.
    - Y_k_new: Target data point generated from the system.
    - learning_rate: The learning rate for gradient descent.
    
    Returns:
    - A_k: Updated matrix after the gradient descent step.
    
    Notes:
    - The gradient is calculated based on the difference between the predicted value (A_k @ X_k_new) 
      and the target value (Y_k_new).
    - The update step adjusts A_k by moving in the direction opposite to the gradient.
    """
    n, _ = X_k_new.shape
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T
    A_k -= learning_rate * grad 
    return A_k


def run_online_gradient_descent(A_k, D, X_k_inital,  learning_rate, iterations):
    """
    Run online gradient descent to update matrix A_k using generated data.
          
    Parameters:
    - A_k: Initial matrix A_k to be updated.
    - D: The matrix used to generate Y_k_new.
    - lambda_reg: The regularization parameter.
    - learning_rate: The learning rate for gradient descent.
    - iterations: Number of iterations to run the gradient descent.
    
    Returns:
    - A_k_optimized: Updated matrix A_k after gradient descent.
    - frobenius_diffs: List of Frobenius norm differences at each iteration.
    """
    n = A_k.shape[0]
    frobenius_diffs = []
    X_k = X_k_inital
    # X_k, Y_k= generate_random_data(D, X_k, n)
    # A_k = odmd.update(X_k.flatten(), Y_k.flatten()) 
    for _ in range(iterations):
        # print(_)
        X_k = np.random.randn(n, 1)
        X_k, X_k_new = generate_random_data(D, X_k, n)
        A_k = gradient_descent_update(A_k, X_k, X_k_new , learning_rate)
        
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)
        X_k = np.random.randn(n, 1)
    return A_k, frobenius_diffs



# Set parameters
n = 10  # Dimension of the matrices and vectors
learning_rate = 0.01  # Learning rate for gradient descent
iterations = 20000  # Number of iterations for gradient descent
D = np.random.randn(n, n) * 1.5  # Generate a random matrix D
A_k = np.zeros((n, n))  # Initialize matrix A_k with zeros
X_k_inital = np.random.randn(n, 1)  # Generate an initial random state vector

# Run online gradient descent
# Perform online gradient descent for different types of matrix constraints
triangular_matrix, frobenius_diffs = run_online_gradient_descent_triangular(A_k, D, X_k_inital, learning_rate, iterations)
circulant_matrix, frobenius_diffs = run_online_gradient_descent_circulant(A_k, D, X_k_inital, learning_rate, iterations)
tridiagonal_matrix, frobenius_diffs = run_online_gradient_descent_Tri_Diagonal(A_k, D, X_k_inital, learning_rate, iterations)
symmetric_matrix, frobenius_diffs = run_online_gradient_descent_symmetric(A_k, D, X_k_inital, learning_rate, iterations)
A_k_optimized, frobenius_diffs = run_online_gradient_descent(A_k, D, X_k_inital, learning_rate, iterations)

# Generate data for piDMD
X, Y = generate_data_matrix(D, 10, 20000)

# Run piDMD
# Perform PiDMD for different types of matrix constraints
triangular_matrix_pi = PiDMD(manifold="uppertriangular", compute_A=True).fit(X, Y)
symmetric_matrix_pi = PiDMD(manifold="symmetric", compute_A=True).fit(X, Y)
circulant_matrix_pi = PiDMD(manifold="circulant", compute_A=True).fit(X, Y)
tridiagonal_matrix_pi = PiDMD(manifold="diagonal", manifold_opt=2, compute_A=True).fit(X, Y)

# Visualize matrix A
visualize_matrix(A_k_optimized, "Original Matrix")  # Visualize the optimized matrix A
visualize_eigenvalues(A_k_optimized, "Original Matrix")  # Visualize the eigenvalues of the optimized matrix A

# Call the function with the list of matrices and titles
visualize_matrices(
    [circulant_matrix, symmetric_matrix, triangular_matrix, tridiagonal_matrix],
    ['Circulant Matrix', 'Symmetric Matrix', 'Triangular Matrix', 'Tridiagonal Matrix']
)

# Visualize matrices obtained from PiDMD
visualize_matrices(
    [circulant_matrix_pi.A, symmetric_matrix_pi.A, triangular_matrix_pi.A, tridiagonal_matrix_pi.A],
    ['Circulant Matrix', 'Symmetric Matrix', 'Triangular Matrix', 'Tridiagonal Matrix']
)









































