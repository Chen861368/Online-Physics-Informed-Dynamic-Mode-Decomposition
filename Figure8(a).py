# -*- coding: utf-8 -*-
import numpy as np
from gradient_descent_functions import gradient_descent_update_operator_norm
from visualization_functions import visualize_matrix, visualize_eigenvalues, plot_frobenius_diff, plot_gradient_descent_paths
from generate_data import generate_stable_matrix, generate_random_data, generate_sequential_control_data
np.random.seed(5)
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

def save_to_csv_on_desktop(lambda_vals, counts, filename="eigenvalue_counts.csv"):
    # Build the path to the desktop
    desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
    # Create the full path for the CSV
    full_path = os.path.join(desktop_path, filename)
    # Create a DataFrame and save it as a CSV
    df = pd.DataFrame({
        'Lambda': lambda_vals,
        'Eigenvalue_Counts': counts
    })
    df.to_csv(full_path, index=False)
    print(f"Data saved to {full_path}")


def run_online_gradient_descent_operator_norm(A_k, D, lambda_reg, learning_rate, iterations):
    """
    Run online gradient descent to update matrix A_k using generated data.
    
    Parameters:
    - A_k: Initial matrix A_k to be updated.
    - D: The matrix used to generate data.
    - lambda_reg: The regularization parameter.
    - learning_rate: The learning rate for gradient descent.
    - iterations: Number of iterations to run the gradient descent.
    
    Returns:
    - A_k_optimized: Updated matrix A_k after gradient descent.
    """
    n = A_k.shape[0]
    for _ in range(iterations):
        X_k = np.random.randn(n, 1)
        X_k, X_k_new = generate_random_data(D, X_k, n)
        A_k = gradient_descent_update_operator_norm(A_k, X_k, X_k_new , lambda_reg, learning_rate)
    return A_k

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

def plot_eigenvalue_counts(lambda_values, eigenvalue_counts):
    """
    Plot the count of eigenvalues with modulus greater than 1 as a function of lambda in a style suitable for
    high-impact scientific journals such as Nature or Science.

    Parameters:
    - lambda_values: The lambda values used in the gradient descent.
    - eigenvalue_counts: The count of eigenvalues with modulus greater than 1 for each lambda.
    """
    # Use the 'seaborn-colorblind' style for better color visibility and scientific aesthetic
    mpl.style.use('seaborn-colorblind')

    # Creating the plot with specific size and resolution
    plt.figure(figsize=(8, 6), dpi=500)

    # Scatter plot with color and marker adjustments for elegance and visibility
    plt.scatter(lambda_values, eigenvalue_counts, color='blue', alpha=0.75, edgecolors='w', linewidth=0, s=50)
    
    # Enhancing font sizes for clarity and readability
    plt.xlabel('$\lambda$', fontsize=16, fontweight='bold')
    plt.ylabel('Count of Eigenvalues > 1', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Setting title with adjusted font size and font weight
    plt.title('Eigenvalues Count vs $\lambda$', fontsize=18, fontweight='bold')

    # Customizing the legend to match the scientific aesthetic
    plt.legend(['Count of Eigenvalues > 1'], fontsize=14, frameon=True, shadow=True, facecolor='white', edgecolor='black')

    # Enhancing grid visibility and style for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Adding tight layout to ensure the plot is neatly arranged
    plt.tight_layout()

    # Saving the figure with a transparent background for versatility in publication
    plt.savefig('eigenvalues_count_vs_lambda.png', dpi=300, bbox_inches='tight', transparent=True)

    # Displaying the plot
    plt.show()


# Parameters setup
n = 10  # Dimension of the state space
iterations = 10000
learning_rate = 0.03

# Initialize the results dictionary
lambda_values = np.arange(0, 50.51, 0.6)
eigenvalue_counts = []

for lambda_reg in lambda_values:
    A_k = np.zeros((n, n))  # Reinitialize A_k for each lambda
    D = np.random.randn(n, n) * 10  # Assuming D remains constant for simplification
    
    # Run gradient descent
    A_k_optimized = run_online_gradient_descent_operator_norm(A_k, D, lambda_reg, learning_rate, iterations)
    
    # Count eigenvalues greater than one and record
    count = count_eigenvalues_greater_than_one(A_k_optimized)
    eigenvalue_counts.append(count)
    print(lambda_reg )



# Plot the results
plot_eigenvalue_counts(lambda_values, eigenvalue_counts)
save_to_csv_on_desktop(lambda_values, eigenvalue_counts)
