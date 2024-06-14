import numpy as np
from gradient_descent_functions import gradient_descent_update_unclear_norm
from visualization_functions import visualize_matrix, visualize_eigenvalues, plot_frobenius_diff, plot_gradient_descent_paths,visualize_matrix_and_eigenvalues
from generate_data import  generate_stable_matrix, generate_random_data, generate_sequential_control_data
import matplotlib.pyplot as plt
import matplotlib as mpl


np.random.seed(5)

import pandas as pd
import os

def save_rank_data_to_desktop(lambda_values, ranks, filename='rank_data.csv'):
    # Get the path to the user's desktop
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    # Combine the desktop path with the filename
    full_path = os.path.join(desktop_path, filename)
    # Create a DataFrame from the data
    data = pd.DataFrame({
        'Lambda': lambda_values,
        'Rank': ranks
    })
    # Save the DataFrame to a CSV file
    data.to_csv(full_path, index=False)
    print(f"Data saved to {full_path}")


def plot_ranks_vs_lambda(lambda_values, ranks):
    """
    Plot the rank of the matrix A_k as a function of lambda in a style suitable for
    high-impact scientific journals such as Nature or Science, using a scatter plot.

    Parameters:
    - lambda_values: A list of lambda values used in the gradient descent.
    - ranks: A list of ranks of A_k corresponding to each lambda value.
    """
    mpl.style.use('seaborn-colorblind')
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(lambda_values, ranks, color='blue', edgecolor='w', linewidth=0, s=50, alpha=0.75)
    plt.xlabel('$\lambda$', fontsize=16)
    plt.ylabel('Rank of $A_k$', fontsize=16)
    plt.title('Rank of $A_k$ vs $\lambda$', fontsize=18)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('rank_vs_lambda_scatter.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

    
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

# Initialize A_k
n, m = 10, 10  # Dimensions of the state space and observation space
A_k = np.zeros((n, n))  # Initialize matrix A_k with zeros

# Generate data for a low-rank dynamical system
rank = 10  # Rank of the low-rank matrix
D = np.random.randn(n, n) * 1.5  # Generate a random matrix D
# U, _, Vt = np.linalg.svd(D, full_matrices=False)
# D = U[:, :rank] @ Vt[:rank, :]  # Make D a low-rank matrix (uncomment if needed)

# Parameter settings
lambda_reg = 0  # Regularization parameter
learning_rate = 0.03  # Learning rate for gradient descent
iterations = 1000  # Number of iterations for gradient descent


lambda_values = np.arange(0, 1.51, 0.025)
ranks = []

for lambda_reg in lambda_values:
    A_k = np.zeros((n, n))  # Initialize A_k
    A_k_optimized, _, _ = run_online_gradient_descent_unclear_norm(A_k, D, lambda_reg, learning_rate, iterations)
    rank_A_k = np.linalg.matrix_rank(A_k_optimized)
    ranks.append(rank_A_k)

# Plot
plot_ranks_vs_lambda(lambda_values, ranks)
save_rank_data_to_desktop(lambda_values, ranks)










