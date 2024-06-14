# -*- coding: utf-8 -*-
import numpy as np
from gradient_descent_functions import gradient_descent_update_unclear_norm
from visualization_functions import visualize_matrix, visualize_eigenvalues, plot_frobenius_diff, plot_gradient_descent_paths,visualize_matrix_and_eigenvalues
from generate_data import  generate_stable_matrix, generate_random_data, generate_sequential_control_data
import matplotlib.pyplot as plt
import matplotlib as mpl
from gradient_descent_functions import  gradient_descent_update_l2
from gradient_descent_functions import  make_sparse , gradient_descent_update_l1
from gradient_descent_functions import  gradient_descent_update_operator_norm

np.random.seed(5)  # Set the random seed to 5

def run_online_gradient_descent_unclear_norm(A_k, D, lambda_reg, learning_rate, iterations):
    """
    Run online gradient descent to update matrix A_k using generated data.
    
    Parameters:
    - A_k: Initial matrix A_k to be updated.
    - D: The matrix used to generate Y_k_new.
    - lambda_reg: The regularization parameter.
    - learning_rate: The learning rate for gradient descent.
    - iterations: Number of iterations to run the gradient descent.
    
    Returns:
    - A_k: Updated matrix A_k after gradient descent.
    - frobenius_diffs: List of Frobenius norm differences at each iteration.
    """
    n = A_k.shape[0]
    frobenius_diffs = []
    rank_A_ks = []
    for _ in range(iterations):
        X_k = np.random.randn(n, 1)
        X_k_new, Y_k_new = generate_random_data(D, X_k, n)
        A_k = gradient_descent_update_unclear_norm(A_k, X_k_new, Y_k_new, lambda_reg, learning_rate)
        rank_A_k = np.linalg.matrix_rank(A_k)
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)
        rank_A_ks.append(rank_A_k)
    return A_k, frobenius_diffs ,rank_A_ks 


def run_online_gradient_descent_l2(A_k, D, X_k_inital, lambda_reg, learning_rate, iterations):
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
    for _ in range(iterations):
        X_k = np.random.randn(n, 1)
        X_k, X_k_new = generate_random_data(D, X_k, n)
        A_k = gradient_descent_update_l2(A_k, X_k, X_k_new , lambda_reg, learning_rate)
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)
    return A_k, frobenius_diffs

def run_experiments(D, X_k_inital, num_experiments, n, lambda_reg, learning_rate, iterations):
    """
    Run multiple experiments of online gradient descent to observe performance.
    
    Parameters:
    - D: The matrix used to generate Y_k_new.
    - X_k_inital: Initial state vector used to generate data.
    - num_experiments: Number of experiments to run.
    - n: Dimension of the matrices and vectors.
    - lambda_reg: The regularization parameter.
    - learning_rate: The learning rate for gradient descent.
    - iterations: Number of iterations to run the gradient descent in each experiment.
    
    Returns:
    - all_diffs: List of Frobenius norm differences for each experiment.
    """
    all_diffs = []
    for exp in range(num_experiments):
        A_k = np.random.randn(n, n)
        A_k_optimized, frobenius_diffs = run_online_gradient_descent_l2(A_k, D, X_k_inital, lambda_reg, learning_rate, iterations)
        all_diffs.append(frobenius_diffs)
    return all_diffs

def run_online_gradient_descent_l1(A_k, D, lambda_reg, learning_rate, sparsity_threshold, iterations):
    """
    Run online gradient descent with L1 regularization to update matrix A_k using generated data.
    
    Parameters:
    - A_k: Initial matrix A_k to be updated.
    - D: The sparse matrix D used to generate Y_k_new.
    - lambda_reg: The regularization parameter for L1 regularization.
    - learning_rate: The learning rate for gradient descent.
    - sparsity_threshold: The threshold for sparsity in matrix A_k.
    - iterations: Number of iterations to run the gradient descent.
    
    Returns:
    - A_k_sparse: Updated sparse matrix A_k after gradient descent.
    - frobenius_diffs: List of Frobenius norm differences for each iteration.
    """
    n = A_k.shape[0]
    frobenius_diffs = []
    for _ in range(iterations):
        X_k = np.random.randn(n, 1)
        X_k_new, Y_k_new = generate_random_data(D,X_k, n)
        A_k = gradient_descent_update_l1(A_k, X_k_new, Y_k_new, lambda_reg, learning_rate)
        A_k_sparse = make_sparse(A_k, sparsity_threshold)
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)
    return A_k_sparse.toarray(), frobenius_diffs


def count_eigenvalues_greater_than_one(matrix):
    """
    Count the eigenvalues of the matrix that have a modulus greater than 1.
    
    Parameters:
    - matrix: The matrix to calculate the eigenvalues of.
    
    Returns:
    - count: The count of eigenvalues with modulus greater than 1.
    """
    eigenvalues = np.linalg.eigvals(matrix)
    return np.sum(np.abs(eigenvalues) > 1)


def run_online_gradient_descent_operator_norm(A_k, D, X_k_inital, lambda_reg, learning_rate, iterations):
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
    eigenvalue_diffs = []
    X_k = X_k_inital
    for _ in range(iterations):
        X_k = np.random.randn(n, 1)
        X_k, X_k_new = generate_random_data(D, X_k, n)
        A_k = gradient_descent_update_operator_norm(A_k, X_k, X_k_new , lambda_reg, learning_rate)
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)
        eigenvalues = np.linalg.eigvals(A_k)
        eigenvalues_2norms = np.abs(eigenvalues)
        eigenvalue_diff = np.sum(np.abs(eigenvalues_2norms - 1))
        eigenvalue_diffs.append(eigenvalue_diff)
    return A_k, frobenius_diffs, eigenvalue_diffs


def visualize_matrices(matrices, titles, save_path="C:\\Users\\HIT\\Desktop"):
    num_matrices = len(matrices)
    fig, axs = plt.subplots(1, num_matrices, figsize=(num_matrices * 5, 5))  # Adjust subplot size based on number of matrices

    # Find global maximum for color range normalization across all matrices
    vmax = max([np.abs(matrix).max() for matrix in matrices])
    vmin = -vmax
    
    # Set the font to Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.size'] = 22

    for i, (matrix, title) in enumerate(zip(matrices, titles)):
        ax = axs[i]  # Get the subplot for this matrix
        cax = ax.matshow(matrix, cmap="seismic", vmax=vmax, vmin=vmin)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)  # Normalize colorbar size
        ax.set_title(title, y=-0.15)  # Adjust title position
        ax.grid(visible=False)  # Hide grid lines
        ax.tick_params(axis='both', which='both', length=0)  # Hide ticks
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(wspace=0.2, hspace=0)  # Adjust the spacing between subplots
    plt.savefig(f"{save_path}\\matrices_visualization.pdf", format='pdf', bbox_inches='tight')
    plt.show()

def visualize_eigenvalues_of_matrices(matrices, titles, save_path="C:\\Users\\HIT\\Desktop"):
    num_matrices = len(matrices)
    fig, axs = plt.subplots(1, num_matrices, figsize=(5 * num_matrices, 5), dpi=300)

    # Set the font to Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.size'] = 21

    # Determine common axis limits for all subplots based on the eigenvalues range
    all_eigs = np.concatenate([np.linalg.eigvals(matrix) for matrix in matrices])
    lim = np.max(np.abs(all_eigs))
    limits = (-lim, lim)

    for i, (matrix, title) in enumerate(zip(matrices, titles)):
        eigs = np.linalg.eigvals(matrix)
        ax = axs[i]
        ax.axvline(x=0, c="k", lw=1)
        ax.axhline(y=0, c="k", lw=1)
        t = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(t), np.sin(t), c="gray", ls="--")  # Unit circle
        ax.scatter(eigs.real, eigs.imag, c="blue", alpha=0.7)  # Eigenvalues
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_xlabel("Real")
        ax.set_ylabel("Imag")
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
        ax.tick_params(axis='both', which='both', length=0)  # Hide ticks
        ax.set_xticks([])
        ax.set_yticks([])
    plt.subplots_adjust(wspace=0.12, hspace=0)  # Adjust the spacing between subplots
    plt.savefig(f"{save_path}\\eigenvalues_visualization.pdf", format='pdf', bbox_inches='tight')
    plt.show()




# Initialize A_k
n, m = 10, 10  # Dimensions of the state space and observation space
A_k = np.zeros((n, n))
D = np.random.randn(n, n) * 1.5
X_k_inital = np.random.randn(n, 1)

# Parameter settings
lambda_reg_unclear = 0.22  # Regularization parameter for unclear norm
lambda_reg_l1 = 0.06  # Regularization parameter for L1 regularization
lambda_reg_l2 = 18  # Regularization parameter for L2 regularization
lambda_reg_operator = 0.6  # Regularization parameter for operator norm

learning_rate = 0.01  # Learning rate for gradient descent
iterations = 10000  # Number of iterations for gradient descent
sparsity_threshold = 0  # Threshold for making the matrix sparse

# Run online gradient descent
Low_rank_matrix, frobenius_diffs, all_ranks = run_online_gradient_descent_unclear_norm(A_k, D, lambda_reg_unclear, learning_rate, iterations)
Stable_matrix, frobenius_diffs, eigenvalue_diff = run_online_gradient_descent_operator_norm(A_k, D, X_k_inital, lambda_reg_operator, learning_rate, iterations)
Norm_shrinkage_matrix, frobenius_diffs = run_online_gradient_descent_l2(A_k, D, X_k_inital, lambda_reg_l2, learning_rate, iterations)
Sparse_matrix, frobenius_diffs = run_online_gradient_descent_l1(A_k, D, lambda_reg_l1, learning_rate, sparsity_threshold, iterations)


# Call the function with the list of matrices and titles
visualize_matrices(
    [Stable_matrix,Low_rank_matrix , Sparse_matrix, Norm_shrinkage_matrix],
    ['Stable Matrix','Low rank Matrix', 'Sparse Matrix','Norm shrinkage Matrix']
)

visualize_eigenvalues_of_matrices(
    [Stable_matrix, Low_rank_matrix, Sparse_matrix, Norm_shrinkage_matrix],
    ['Stable Matrix', 'Low rank Matrix', 'Sparse Matrix', 'Norm shrinkage Matrix']
)































