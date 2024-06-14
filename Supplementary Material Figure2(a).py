# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pydmd import PiDMD
import time
from sklearn.preprocessing import StandardScaler
np.random.seed(0)

def run_pidmd_experiment(dimensions, num_samples, constraints):
    """
    Run the PiDMD experiment with various constraints and dimensions.
    
    Args:
    - dimensions: List of integers representing dimensions to test.
    - num_samples: Integer, number of samples (data points) for each dimension.
    - constraints: List of strings, each representing a matrix constraint.
    
    Returns:
    - results: Dict, keys are constraints and values are lists of times.
    """
    results = {constraint: [] for constraint in constraints}
    for dim in dimensions:
        print(f"Dimension: {dim}")
        X = np.random.standard_normal((dim, num_samples))*10
        noise = np.random.randn(dim, num_samples)
        X = X + noise
        for constraint in constraints:
            start_time = time.time()
            if constraint != "tridiagonal":
                pidmd = PiDMD(manifold=constraint, compute_A=True).fit(X)
            else:
                pidmd = PiDMD(manifold="diagonal", manifold_opt=2, compute_A=True).fit(X)
            elapsed_time = time.time() - start_time
            results[constraint].append(elapsed_time)
    
    return results

def plot_results(dimensions, results):
    """
    Plot the experiment results.
    
    Args:
    - dimensions: List of dimensions tested.
    - results: Dict, optimization times for each constraint.
    """
    plt.figure(figsize=(6, 5), dpi=300)
    for constraint, times in results.items():
        plt.scatter(dimensions, times, alpha=0.7, label=f'{constraint}')
    
    plt.xlabel('n (Dimension)', fontsize=14, fontweight='bold')
    plt.ylabel('Time (Seconds)', fontsize=14, fontweight='bold')
    plt.title('Physics Informed DMD Optimization Time ', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

import pandas as pd
import os

def save_results_to_desktop(dimensions, results, filename='pidmd_experiment_results.csv'):
    """
    Save the PiDMD experiment results to a CSV file on the desktop.

    Parameters:
    - dimensions: List of dimensions tested in the experiment.
    - results: Dictionary with constraints as keys and list of times as values.
    - filename: The name of the file to save the results.
    """
    # Build the path to the desktop
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    full_path = os.path.join(desktop_path, filename)
    
    # Prepare the data for DataFrame
    data = []
    for constraint, times in results.items():
        for dim, time in zip(dimensions, times):
            data.append({'Dimension': dim, 'Constraint': constraint, 'Time': time})
    
    # Convert the data into a DataFrame
    df = pd.DataFrame(data)
    
    # Save the DataFrame as a CSV file
    df.to_csv(full_path, index=False)
    print(f"Results saved to {full_path}")




# Define experiment parameters
dimensions = range(100, 1001, 100)  # From 500 to 1500, stepping by 100
num_samples = 10000
constraints = ['circulant', 'symmetric', 'uppertriangular', 'tridiagonal']

# Run the experiment
results = run_pidmd_experiment(dimensions, num_samples, constraints)

# Plot the results
plot_results(dimensions, results)


# After running your experiments and obtaining the `results` dictionary:
save_results_to_desktop(dimensions, results)






