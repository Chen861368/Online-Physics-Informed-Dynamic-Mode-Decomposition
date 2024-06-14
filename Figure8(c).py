# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse import rand
import random
from gradient_descent_functions import  make_sparse , gradient_descent_update_l1
from visualization_functions import visualize_matrix, visualize_eigenvalues, plot_frobenius_diff, plot_gradient_descent_paths
from generate_data import  generate_stable_sparse_matrix, generate_random_data
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.sparse import coo_matrix
import scipy

np.random.seed(5)
import pandas as pd
import os

def save_nonzero_data_to_desktop(lambda_values, non_zero_counts, filename='nonzero_elements_data.csv'):
    # Get the path to the user's desktop
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    # Combine the desktop path with the filename
    full_path = os.path.join(desktop_path, filename)
    # Create a DataFrame from the lambda_values and non_zero_counts
    data = pd.DataFrame({
        'Lambda': lambda_values,
        'Non_Zero_Counts': non_zero_counts
    })
    # Save the DataFrame to a CSV file
    data.to_csv(full_path, index=False)
    print(f"Data saved to {full_path}")

# Example call to the function

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
        # frobenius_diff = np.linalg.norm(A_k - D.toarray(), 'fro')
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)
    return A_k_sparse, frobenius_diffs


def count_nonzero_elements(matrix):
    """
    Count the number of non-zero elements in the matrix.
    Works for both dense matrices (numpy arrays) and sparse matrices (scipy.sparse).
    
    Parameters:
    - matrix: The matrix to be checked.
    
    Returns:
    - count: The count of non-zero elements.
    """
    # If the matrix is a SciPy sparse matrix, use the .nnz attribute
    if scipy.sparse.issparse(matrix):
        return matrix.nnz
    else:
        # If the matrix is a dense NumPy array, use np.count_nonzero
        return np.count_nonzero(matrix)

def plot_nonzero_elements_vs_lambda(lambda_values, non_zero_counts):
    """
    Plot the number of non-zero elements of the matrix A_k as a function of lambda in a scatter plot format,
    with a style suitable for high-impact scientific journals such as Nature or Science.

    Parameters:
    - lambda_values: A list of lambda values used in the gradient descent.
    - non_zero_counts: A list of counts of non-zero elements of A_k corresponding to each lambda value.
    """
    # Use a simple, clean style for the plot
    mpl.style.use('seaborn-whitegrid')
    
    # Create a figure with a higher resolution for publication quality
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Plot data as a scatter plot
    plt.scatter(lambda_values, non_zero_counts, c='blue', s=40, alpha=0.7)
    
    # Set the axis labels with a larger font for readability
    plt.xlabel('$\lambda$', fontsize=16)
    plt.ylabel('Number of Non-Zero Elements', fontsize=16)
    
    # Set the title of the plot
    plt.title('Non-Zero Elements in $A_k$ vs $\lambda$', fontsize=18)
    
    # Enhance grid visibility for better readability, consistent with scientific publications
    plt.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.7)
    
    # Remove the top and right spines for a cleaner look
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Apply a tight layout to ensure all labels and titles are visible
    plt.tight_layout()
    
    # Show the plot
    plt.show()

# Setting up the experiment
lambda_values = np.arange(0, 0.8, 0.015)  # Range of lambda values for regularization
non_zero_counts = []  # List to store counts of non-zero elements for each lambda

n = 10  # Dimension of the matrices
D = np.random.randn(n, n)  # Generate a random matrix D
learning_rate = 0.03  # Learning rate for gradient descent
sparsity_threshold = 0  # Threshold for making the matrix sparse
iterations = 10000  # Number of iterations for gradient descent


for lambda_reg in lambda_values:
    print(lambda_reg)
    # Reset A_k to zero matrix for each run
    A_k = np.zeros((n, n))
    # Run gradient descent
    A_k_optimized, _ = run_online_gradient_descent_l1(A_k, D, lambda_reg, learning_rate, sparsity_threshold, iterations)
    # Count non-zero elements
    count_nonzero = count_nonzero_elements(A_k_optimized)
    non_zero_counts.append(count_nonzero)

# Plot the results
plot_nonzero_elements_vs_lambda(lambda_values, non_zero_counts)
save_nonzero_data_to_desktop(lambda_values, non_zero_counts)




