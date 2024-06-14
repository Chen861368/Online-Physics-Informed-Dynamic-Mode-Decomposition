# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from gradient_descent_functions import  gradient_descent_update_l2
from visualization_functions import visualize_matrix, visualize_eigenvalues, plot_frobenius_diff, plot_gradient_descent_paths,compute_and_plot_singular_values_scatter
from generate_data import  generate_stable_matrix, generate_random_data, generate_sequential_control_data

np.random.seed(5)

# Assuming gradient_descent_update_l2 and generate_random_data functions are defined correctly.
# These should be included in gradient_descent_functions.py and generate_data.py respectively.

def run_online_gradient_descent_l2(A_k, D, X_k_inital, lambda_reg, learning_rate, iterations):
    n = A_k.shape[0]
    frobenius_diffs = []
    X_k = X_k_inital
    for _ in range(iterations):
        X_k = np.random.randn(n, 1)
        X_k, X_k_new = generate_random_data(D, X_k, n)
        A_k = gradient_descent_update_l2(A_k, X_k, X_k_new, lambda_reg, learning_rate)
        # frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        # frobenius_diffs.append(frobenius_diff)
    return A_k, frobenius_diffs

def run_experiment_and_plot(D, X_k_initial, learning_rate, iterations):
    lambda_values = np.arange(0, 15.51, 0.1)
    frobenius_norms = []
    
    for lambda_reg in lambda_values:
        print(lambda_reg)
        A_k = np.zeros((n, n))
        A_k_optimized, _ = run_online_gradient_descent_l2(A_k, D, X_k_initial, lambda_reg, learning_rate, iterations)
        frobenius_norm = np.linalg.norm(A_k_optimized, 'fro')
        frobenius_norms.append(frobenius_norm)
    
    save_data_to_csv(lambda_values, frobenius_norms, 'frobenius_vs_lambda.csv')
    plot_frobenius_vs_lambda(lambda_values, frobenius_norms)

def plot_frobenius_vs_lambda(lambda_values, frobenius_norms):
    mpl.style.use('seaborn-colorblind')
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(lambda_values, frobenius_norms, color='blue', edgecolor='w', linewidth=0, s=50, alpha=0.75)
    plt.xlabel('$\lambda$', fontsize=16)
    plt.ylabel('Final Frobenius Norm', fontsize=16)
    plt.title('Frobenius Norm of $A_k$ vs $\lambda$', fontsize=18)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('frobenius_vs_lambda.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

def save_data_to_csv(lambda_values, frobenius_norms, filename):
    # Build the path to the desktop based on the current user's home directory
    desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
    full_path = os.path.join(desktop_path, filename)
    
    # Convert data to pandas DataFrame
    df = pd.DataFrame({
        'Lambda': lambda_values,
        'Frobenius_Norm': frobenius_norms
    })
    
    # Save DataFrame as a CSV file
    df.to_csv(full_path, index=False)
    print(f"Data saved to {full_path}")

# Define matrices and parameters for the experiments
n = 10  # Dimension of the state space
learning_rate = 0.03
iterations = 10000
D = np.random.rand(n, n)  # Random matrix D
X_k_initial = np.random.randn(n, 1)  # Initial state X_k

# Run the experiment and plot the results
run_experiment_and_plot(D, X_k_initial, learning_rate, iterations)


