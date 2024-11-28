# -*- coding: utf-8 -*-
import numpy as np
import time
import matplotlib.pyplot as plt
np.random.seed(0)


def project_to_cyclic_matrix(matrix):
    """
    Efficiently project a matrix to a cyclic matrix based on its first row,
    avoiding issues with np.roll for non-scalar shift values.

    Parameters:
    - matrix: A NumPy array representing the original matrix.

    Returns:
    - A NumPy array representing the projected cyclic matrix.
    """
    first_row = matrix[0, :]
    n = first_row.size

    # Initialize the cyclic matrix
    cyclic_matrix = np.zeros_like(matrix)

    # Manually create the cyclic shifts
    for i in range(n):
        cyclic_matrix[i, :] = np.concatenate((first_row[-i:], first_row[:-i]))

    return cyclic_matrix


def gradient_descent_update_circulant(A_k, X_k_new, Y_k_new, learning_rate):
    """
    Perform a single update step using online gradient descent with a circulant matrix constraint.
    
    Parameters:
    - A_k: Current matrix to be updated.
    - X_k_new: New input data.
    - Y_k_new: New output data.
    - learning_rate: The learning rate for gradient descent.
    
    Returns:
    - A_k: Updated circulant matrix.
    """
    n, _ = X_k_new.shape
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T
    A_k -= learning_rate * grad
    return project_to_cyclic_matrix(A_k)

def projection_onto_upper_triangular(A):
    """
    Project a matrix onto the space of upper triangular matrices.
    
    Parameters:
    - A: The matrix to be projected.
    
    Returns:
    - A: The projected upper triangular matrix.
    """
    return np.triu(A)

def gradient_descent_update_triangular(A_k, X_k_new, Y_k_new, learning_rate):
    """
    Perform a single update step using online gradient descent with an upper triangular matrix constraint.
    
    Parameters:
    - A_k: Current matrix to be updated.
    - X_k_new: New input data.
    - Y_k_new: New output data.
    - learning_rate: The learning rate for gradient descent.
    
    Returns:
    - A_k: Updated upper triangular matrix.
    """
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T 
    A_k -= learning_rate * grad
    A_k = projection_onto_upper_triangular(A_k)  # Apply the upper triangular matrix constraint
    return A_k

def projection_onto_tridiagonal(A):
    """
    Efficiently project a matrix onto the space of tridiagonal matrices.
    
    Parameters:
    - A: The matrix to be projected.
    
    Returns:
    - A: The projected tridiagonal matrix.
    """
    # Use NumPy's triu (upper triangle) and tril (lower triangle) functions
    # to keep only the main diagonal and the first diagonals above and below it.
    return np.triu(np.tril(A, 1), -1)

def gradient_descent_update_tridiagonal(A_k, X_k_new, Y_k_new, learning_rate):
    """
    Perform a single update step using online gradient descent with a tridiagonal matrix constraint.
    
    Parameters:
    - A_k: Current matrix to be updated.
    - X_k_new: New input data.
    - Y_k_new: New output data.
    - learning_rate: The learning rate for gradient descent.
    
    Returns:
    - A_k: Updated tridiagonal matrix.
    """
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T 
    A_k -= learning_rate * grad
    A_k = projection_onto_tridiagonal(A_k)  # Apply the tridiagonal matrix constraint
    return A_k

def projection_onto_symmetric(A):
    """
    Project a matrix onto the space of symmetric matrices.
    
    Parameters:
    - A: The matrix to be projected.
    
    Returns:
    - A: The projected symmetric matrix.
    """
    return (A + A.T) / 2

def gradient_descent_update_symmetric(A_k, X_k_new, Y_k_new, learning_rate):
    """
    Perform a single update step using online gradient descent with a symmetric matrix constraint.
    
    Parameters:
    - A_k: Current matrix to be updated.
    - X_k_new: New input data.
    - Y_k_new: New output data.
    - learning_rate: The learning rate for gradient descent.
    
    Returns:
    - A_k: Updated symmetric matrix.
    """
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T 
    A_k -= learning_rate * grad
    A_k = projection_onto_symmetric(A_k)  # Apply the symmetric matrix constraint
    return A_k



def gradient_descent_update(A_k, X_k_new, Y_k_new, learning_rate):
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T
    A_k -= learning_rate * grad 
    return A_k


def generate_random_data(D, X_k, n):
    Y_k = D @ X_k
    return X_k, Y_k



def run_online_gradient_descent_timed(A_k, D, X_k_initial, learning_rate, iterations):
    start_time = time.time()
    n = A_k.shape[0]
    X_k = X_k_initial
    for _ in range(iterations):
        X_k = np.random.randn(n, 1)
        X_k, X_k_new = generate_random_data(D, X_k, n)
        A_k = gradient_descent_update(A_k, X_k, X_k_new, learning_rate)
    end_time = time.time()
    return (end_time - start_time)/iterations


def run_online_gradient_descent_update_circulant_timed(A_k, D, X_k_initial, learning_rate, iterations):
    start_time = time.time()
    n = A_k.shape[0]
    X_k = X_k_initial
    for _ in range(iterations):
        X_k = np.random.randn(n, 1)
        X_k, X_k_new = generate_random_data(D, X_k, n)
        A_k = gradient_descent_update_circulant(A_k, X_k, X_k_new, learning_rate)
    end_time = time.time()
    return (end_time - start_time)/iterations



def run_online_gradient_descent_update_triangular_timed(A_k, D, X_k_initial, learning_rate, iterations):
    start_time = time.time()
    n = A_k.shape[0]
    X_k = X_k_initial
    for _ in range(iterations):
        X_k = np.random.randn(n, 1)
        X_k, X_k_new = generate_random_data(D, X_k, n)
        A_k = gradient_descent_update_triangular(A_k, X_k, X_k_new, learning_rate)
    end_time = time.time()
    return (end_time - start_time)/iterations


def run_online_gradient_descent_update_tridiagonal_timed(A_k, D, X_k_initial, learning_rate, iterations):
    start_time = time.time()
    n = A_k.shape[0]
    X_k = X_k_initial
    for _ in range(iterations):
        X_k = np.random.randn(n, 1)
        X_k, X_k_new = generate_random_data(D, X_k, n)
        A_k = gradient_descent_update_tridiagonal(A_k, X_k, X_k_new, learning_rate)
    end_time = time.time()
    return (end_time - start_time)/iterations


def run_online_gradient_descent_update_symmetric_timed(A_k, D, X_k_initial, learning_rate, iterations):
    start_time = time.time()
    n = A_k.shape[0]
    X_k = X_k_initial
    for _ in range(iterations):
        X_k = np.random.randn(n, 1)
        X_k, X_k_new = generate_random_data(D, X_k, n)
        A_k = gradient_descent_update_symmetric(A_k, X_k, X_k_new, learning_rate)
    end_time = time.time()
    return (end_time - start_time)/iterations


def plot_optimization_times_for_constraints(n_values, gradient_times, circulant_times, symmetric_times, triangular_times, tridiagonal_times):
    """
    Plot the optimization times for different matrix constraints across varying dimensions.

    Parameters:
    - n_values: List of n (dimension) values.
    - gradient_times: List of times for basic Gradient Descent updates.
    - circulant_times: List of times for Gradient Descent with Circulant matrix constraint.
    - symmetric_times: List of times for Gradient Descent with Symmetric matrix constraint.
    - triangular_times: List of times for Gradient Descent with Upper Triangular matrix constraint.
    - tridiagonal_times: List of times for Gradient Descent with Tri-Diagonal matrix constraint.
    """
    plt.figure(figsize=(6, 5), dpi=300)
    
    plt.scatter(n_values, gradient_times, color='red', alpha=0.7, label='Gradient descent', marker='o')
    plt.scatter(n_values, circulant_times, color='blue', alpha=0.7, label='Circulant matrix', marker='^')
    plt.scatter(n_values, symmetric_times, color='green', alpha=0.7, label='Symmetric matrix', marker='s')
    plt.scatter(n_values, triangular_times, color='purple', alpha=0.7, label='Upper Triangular matrix', marker='p')
    plt.scatter(n_values, tridiagonal_times, color='orange', alpha=0.7, label='Tri-Diagonal matrix', marker='*')
    
    plt.xlabel('n (Dimension)', fontsize=14)
    plt.ylabel('Time (Seconds)', fontsize=14)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plt.show()


learning_rate = 0.0000000004
iterations = 5
n_values = range(100, 1001, 100)  # Reduced range for quicker execution
online_dmd_times = []
gradient_descent_times = []
gradient_descent_circulant_times = []
gradient_descent_triangular_times = []
gradient_descent_tridiagonal_times = []
gradient_descent_symmetric_times = []

for n in n_values:
    print(n)
    D = np.random.rand(n, n)
    A_k = np.zeros((n, n))
    X_k_initial = np.random.randn(n, 1)
    time_taken = run_online_gradient_descent_timed(A_k, D, X_k_initial, learning_rate, iterations)
    gradient_descent_times.append(time_taken)



for n in n_values:
    print(n)
    D = np.random.rand(n, n)
    A_k = np.zeros((n, n))
    X_k_initial = np.random.randn(n, 1)
    time_taken = run_online_gradient_descent_update_circulant_timed(A_k, D, X_k_initial, learning_rate, iterations)
    gradient_descent_circulant_times.append(time_taken)


for n in n_values:
    print(n)
    D = np.random.rand(n, n)
    A_k = np.zeros((n, n))
    X_k_initial = np.random.randn(n, 1)
    time_taken = run_online_gradient_descent_update_tridiagonal_timed(A_k, D, X_k_initial, learning_rate, iterations)
    gradient_descent_tridiagonal_times.append(time_taken)


for n in n_values:
    print(n)
    D = np.random.rand(n, n)
    A_k = np.zeros((n, n))
    X_k_initial = np.random.randn(n, 1)
    time_taken = run_online_gradient_descent_update_symmetric_timed(A_k, D, X_k_initial, learning_rate, iterations)
    gradient_descent_symmetric_times.append(time_taken)



for n in n_values:
    print(n)
    D = np.random.rand(n, n)
    A_k = np.zeros((n, n))
    X_k_initial = np.random.randn(n, 1)
    time_taken = run_online_gradient_descent_update_triangular_timed(A_k, D, X_k_initial, learning_rate, iterations)
    gradient_descent_triangular_times.append(time_taken)



plot_optimization_times_for_constraints(n_values, gradient_descent_times, gradient_descent_circulant_times, gradient_descent_symmetric_times, gradient_descent_triangular_times , gradient_descent_tridiagonal_times)

import pandas as pd
import os

def save_times_to_desktop(n_values, times, labels, filename='optimization_times.csv'):
    """
    Save the optimization times for various constraints to a CSV file on the desktop.

    Parameters:
    - n_values: List of n (dimension) values.
    - times: List of lists, where each list is the optimization times for a specific constraint.
    - labels: List of strings, labels for each set of times in `times`.
    - filename: String, name of the file to save.
    """
    # Build the path to the desktop
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    full_file_path = os.path.join(desktop_path, filename)

    # Prepare data for DataFrame
    data = []
    for label, time_list in zip(labels, times):
        for n, time in zip(n_values, time_list):
            data.append({'Dimension': n, 'Constraint': label, 'Time': time})

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(full_file_path, index=False)
    print(f"Optimization times saved to {full_file_path}")

# Assuming the lists `n_values`, `gradient_descent_times`, `gradient_descent_circulant_times`,
# `gradient_descent_symmetric_times`, `gradient_descent_triangular_times`, and `gradient_descent_tridiagonal_times`
# are already defined and populated with your experiment's data

labels = ['Gradient', 'Circulant', 'Symmetric', 'Upper Triangular', 'Tri-Diagonal']
times = [gradient_descent_times, gradient_descent_circulant_times, gradient_descent_symmetric_times, gradient_descent_triangular_times, gradient_descent_tridiagonal_times]

# Call the function to save the data
save_times_to_desktop(n_values, times, labels)




