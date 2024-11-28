# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time

# Gradient Descent update function optimized for vector operations
def gradient_descent_update_optimized(A_k, X_k_new, Y_k_new, D, learning_rate):
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T
    A_k -= learning_rate * grad
    frobenius_diff = np.linalg.norm(A_k - D, 'fro')
    return A_k, frobenius_diff

# Function to find the optimal learning rate for gradient descent
def find_optimal_learning_rate_optimized(D, X_k_initial, iterations, target_diff=0.1, lr_start=0.01, lr_end=0.0001, lr_step=0.0001):
    optimal_lr = None
    min_frobenius_diff = np.inf
    optimal_time = np.inf
    final_frobenius_diff = None

    # 正确生成从 lr_start 到 lr_end 的学习率序列，包含 lr_end
    learning_rates = np.arange(lr_start, lr_end - lr_step, -lr_step)

    for learning_rate in learning_rates:
        # print(learning_rate)
        A_k = np.zeros((D.shape[0], D.shape[0]))
        start_time = time.time()
        for _ in range(iterations):
            X_k = np.random.randn(D.shape[0], 1)
            Y_k = D @ X_k
            A_k, frobenius_diff = gradient_descent_update_optimized(A_k, X_k, Y_k, D, learning_rate)
            if frobenius_diff < target_diff:
                time_taken = time.time() - start_time
                optimal_lr = learning_rate
                optimal_time = time_taken
                final_frobenius_diff = frobenius_diff
                return optimal_time, optimal_lr, final_frobenius_diff,_
            if frobenius_diff > 700:
                # print(f"learning rate for n={n}: {learning_rate}, Frobenius difference: {frobenius_diff}")
                break  # 如果差异过大，提前结束这个学习率的测试
    return optimal_time, optimal_lr, final_frobenius_diff

# Online DMD class definition
class OnlineDMD:
    def __init__(self, n: int, weighting: float = 0.9) -> None:
        self.n = n
        self.weighting = weighting
        self.timestep = 0
        self.A = np.zeros((n, n))
        self._P = np.zeros((n, n))
        self._initialize()

    def _initialize(self) -> None:
        epsilon = 1e-15
        alpha = 1.0 / epsilon
        self.A = np.random.randn(self.n, self.n)
        self._P = alpha * np.identity(self.n)

    def update(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        Px = self._P.dot(x)
        gamma = 1.0 / (1 + x.T.dot(Px))
        self.A += np.outer(gamma * (y - self.A.dot(x)), Px)
        self._P = (self._P - gamma * np.outer(Px, Px)) / self.weighting
        self._P = (self._P + self._P.T) / 2
        self.timestep += 1
        return self.A

# Function to run online DMD with Frobenius target
def run_online_dmd_with_frobenius_target(D, X_k_initial, iterations, target_diff):
    n = D.shape[0]
    A_k = np.zeros((n, n))
    odmd = OnlineDMD(n, 1)
    X_k = X_k_initial
    start_time = time.time() 
    for i in range(iterations):
        X_k_new = np.random.randn(n, 1)
        Y_k_new = D @ X_k_new
        A_k = odmd.update(X_k_new.flatten(), Y_k_new.flatten())
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        if frobenius_diff < target_diff:
            return time.time() - start_time
    return None

def plot_performance_comparison(n_values, gradient_descent_times, online_dmd_times):
    plt.figure(figsize=(10, 7), dpi=300)
    plt.scatter(n_values, gradient_descent_times, color='red', label='Gradient Descent')
    plt.scatter(n_values, online_dmd_times, color='blue', label='Online DMD')
    plt.xlabel('n (Dimension)', fontsize=14, fontweight='bold')
    plt.ylabel('Time to reach Frobenius norm diff < 0.1', fontsize=14, fontweight='bold')
    plt.title('Performance Comparison', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# Main logic to combine and plot results
n_values = range(100, 1001, 100)
iterations = 25000
target_diff = 0.1

# Initialize lists for storing results
gradient_descent_times = []
online_dmd_times = []

# Initialize the previous optimal learning rate for gradient descent
prev_optimal_lr = 0.01

# Process and collect time measurements for both methods
for n in n_values:
    D = np.random.rand(n, n)
    X_k_initial = np.random.randn(n, 1)
    
    # Online DMD
    dmd_time = run_online_dmd_with_frobenius_target(D, X_k_initial, iterations, target_diff)
    print(f"Online DMD processing n={n}, Optimal time: {dmd_time}")
    online_dmd_times.append(dmd_time if dmd_time is not None else np.nan)
    
    # Gradient Descent
    grad_time, learning_rate, final_frobenius_diff, _ = find_optimal_learning_rate_optimized(D, X_k_initial, iterations, target_diff=0.1, lr_start=prev_optimal_lr)
    print(f"Optimal gradient descent time for n={n}: {grad_time}, learning rate: {learning_rate}, Frobenius difference: {final_frobenius_diff}, Iteration number: {_}")
    gradient_descent_times.append(grad_time)
    prev_optimal_lr = learning_rate  # Update the learning rate for the next iteration
    

# Plotting the performance comparison
plot_performance_comparison(list(n_values), gradient_descent_times, online_dmd_times)



import pandas as pd
import os

def save_performance_data_to_desktop(n_values, gradient_descent_times, online_dmd_times, filename='performance_comparison_data.csv'):
    """
    Save the performance data (time to reach target Frobenius norm difference) for Gradient Descent and Online DMD
    to a CSV file on the desktop.

    Parameters:
    - n_values: List of dimensions (n) tested.
    - gradient_descent_times: List of times taken by Gradient Descent to reach the target Frobenius norm difference.
    - online_dmd_times: List of times taken by Online DMD to reach the target Frobenius norm difference.
    - filename: Name of the CSV file to save the data.
    """
    # Construct the full file path to save on desktop
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    full_file_path = os.path.join(desktop_path, filename)
    
    # Prepare the data for saving
    data = {
        'Dimension': n_values,
        'Gradient Descent Time': gradient_descent_times,
        'Online DMD Time': online_dmd_times
    }
    
    # Create a DataFrame and save it as a CSV file
    df = pd.DataFrame(data)
    df.to_csv(full_file_path, index=False)
    print(f"Performance comparison data saved to {full_file_path}")

# Example usage assuming the lists `n_values`, `gradient_descent_times`, and `online_dmd_times`
# are already populated with data from the performance comparison
save_performance_data_to_desktop(n_values, gradient_descent_times, online_dmd_times)































