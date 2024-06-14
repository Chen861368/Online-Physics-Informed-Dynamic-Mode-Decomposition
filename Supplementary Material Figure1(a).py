# -*- coding: utf-8 -*-
import numpy as np
import time
import matplotlib.pyplot as plt
import numpy.linalg as la
import pandas as pd
import os

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

class StreamingDMD:
    
    """
    Calculate DMD in streaming mode. 
    Python class based on: 
    "Dynamic Mode Decomposition for Large and Streaming Datasets",
    Physics of Fluids 26, 111701 (2014). 
    """

    def __init__(self, max_rank: int =0):
        '''
        Performing Dynamic Mode Decomposition using streaming data.

        Args:
            max_rank: int maximum allowed rank for the linear operator matrix.
        '''

        self.max_rank = max_rank
        self.NGRAM = 5 # Number of Gram_Schmidt iterations
        self.EPSILON = np.finfo(np.float32).eps
        self.Qx = 0
        self.Qy = 0
        self.A = 0
        self.Gx = 0
        self.Gy = 0

    def preprocess(self, x: np.ndarray, y: np.ndarray):
        '''
        Preprocessing step.

        Args:
            x: numpy.ndarray containing the system's state at step i-1
            y: numpy.ndarray containing the system's state at step i
        '''
        
        # Construct bases
        normx = la.norm(x)
        normy = la.norm(y)
        self.Qx = x/normx
        self.Qy = y/normy

        # Compute
        self.Gx = np.zeros([1,1]) + normx**2
        self.Gy = np.zeros([1,1]) + normy**2
        self.A = np.zeros([1,1]) + normx * normy


    def update(self, x: np.ndarray, y: np.ndarray):
        '''
        Updating step.

        Args:
            x: numpy.ndarray containing the system's state at step i-1
            y: numpy.ndarray containing the system's state at step i
        '''

        normx = la.norm(x)
        normy = la.norm(y)     
            
#"       ------- STEP 1 --------       "
        xtilde = np.zeros(shape=(self.Qx.shape[1],1))
        ytilde = np.zeros(shape=(self.Qy.shape[1],1))
        ex = x
        ey = y
        for _ in range(self.NGRAM):
            dx = np.transpose(self.Qx).dot(ex)
            dy = np.transpose(self.Qy).dot(ey)
            xtilde = xtilde + dx
            ytilde = ytilde + dy
            ex = ex - self.Qx.dot(dx)
            ey = ey - self.Qy.dot(dy)
                
#"""       ------- STEP 2 --------       """
#        Check basis for x and expand if needed
        if la.norm(ex) / normx > self.EPSILON:
#           Update basis for x
            self.Qx = np.hstack([self.Qx,ex/la.norm(ex)])

#           Increase size of Gx and A by zero-padding
            self.Gx = np.hstack([self.Gx,np.zeros([self.Gx.shape[0],1])])
            self.Gx = np.vstack([self.Gx,np.zeros([1,self.Gx.shape[1]])])
            self.A  = np.hstack([self.A,np.zeros([self.A.shape[0],1])])
            
        if la.norm(ey) /normy > self.EPSILON:
#           Update basis for y
            self.Qy = np.hstack([self.Qy,ey/la.norm(ey)])

#           Increase size of Gy and A by zero-padding
            self.Gy = np.hstack([self.Gy,np.zeros([self.Gy.shape[0],1])])
            self.Gy = np.vstack([self.Gy,np.zeros([1,self.Gy.shape[1]])])
            self.A  = np.vstack([self.A,np.zeros([1,self.A.shape[1]])]) 
            
#"""       ------- STEP 3 --------       """
#       Check if POD compression is needed
        r0 = self.max_rank
        if r0:
            if self.Qx.shape[1] > r0:
                eigval, eigvec = la.eig(self.Gx)
                indx = np.argsort(-eigval) # get indices for sorting in descending order
                eigval = -np.sort(-eigval) # sort in descending order
                qx = eigvec[:,indx[:r0]]
                self.Qx = self.Qx.dot(qx)
                self.A = self.A.dot(qx)
                self.Gx = np.diag(eigval[:r0])
                
            if self.Qy.shape[1] > r0:
                eigval, eigvec = la.eig(self.Gy)
                indx = np.argsort(-eigval) # get indices for sorting in descending order
                eigval = -np.sort(-eigval) # sort in descending order
                qy = eigvec[:,indx[:r0]]
                self.Qy = self.Qy.dot(qy)
                self.A = np.transpose(qy).dot(self.A)
                self.Gy = np.diag(eigval[:r0])
                
#"""       ------- STEP 4 --------       """
        xtilde = np.transpose(self.Qx).dot(x)
        ytilde = np.transpose(self.Qy).dot(y)
        
#       Update A, Gx and Gy 
        self.A  = self.A + ytilde.dot(np.transpose(xtilde))
        self.Gx = self.Gx + xtilde.dot(np.transpose(xtilde))
        self.Gy = self.Gy + ytilde.dot(np.transpose(ytilde))
        return self.A 
                
def gradient_descent_update(A_k, X_k_new, Y_k_new, learning_rate):
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T
    A_k -= learning_rate * grad 
    return A_k


def generate_random_data(D, X_k, n):
    Y_k = D @ X_k
    return X_k, Y_k


def run_streaming_dmd_update(A_k_initial, D, X_k_initial,  iterations):
    """
    Run online Dynamic Mode Decomposition (DMD) update for matrix A_k using generated data.
    
    Parameters:
    - A_k_initial: Initial DMD matrix A_k to be updated.
    - D: The matrix used to generate Y_k_new.
    - X_k_initial: Initial state vector X_k.
    - lambda_reg: The regularization parameter.
    - learning_rate: The learning rate for gradient update.
    - iterations: Number of iterations to run the update process.
    
    Returns:
    - A_k_optimized: Updated DMD matrix A_k after the update process.
    - frobenius_diffs: List of Frobenius norm differences at each iteration.
    """
    n = A_k_initial.shape[0]
    frobenius_diffs = []
    A_k = A_k_initial
    X_k = X_k_initial
    X_k, Y_k = generate_random_data(D, X_k, n)
    sdmd.preprocess(X_k, Y_k)
    for _ in range(iterations):
        # Generate new data
        X_k, Y_k = generate_random_data(D, X_k, n)
        sdmd.preprocess(X_k, Y_k)
        A_k = sdmd.update(X_k, Y_k)
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)
        X_k = np.random.randn(n, 1)
    return A_k, frobenius_diffs


def run_online_dmd_update_timed(A_k_initial, D, X_k_initial, iterations):
    start_time = time.time()
    odmd = OnlineDMD(n=A_k_initial.shape[0], weighting=1)
    X_k = X_k_initial
    for _ in range(iterations):
        X_k, Y_k = generate_random_data(D, X_k, odmd.n)
        A_k = odmd.update(X_k.flatten(), Y_k.flatten())
        X_k = np.random.randn(odmd.n, 1)
    end_time = time.time()
    return (end_time - start_time)/iterations


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



# Redefining the necessary components to include the requested modifications
def run_streaming_dmd_update_timed(max_rank, D, X_k_initial, iterations):
    """
    Timed version of Streaming DMD update to measure performance over iterations for varying n.
    
    Args:
    - max_rank: Maximum allowed rank of the learned operator.
    - D: The matrix used to generate Y_k_new.
    - X_k_initial: Initial state vector X_k.
    - iterations: Number of iterations to run the update process.
    
    Returns:
    - Time taken for specified number of iterations.
    """
    start_time = time.time()
    n = D.shape[0]
    sdmd = StreamingDMD(max_rank=max_rank)
    X_k = X_k_initial
    X_k, Y_k = generate_random_data(D, X_k, n)
    sdmd.preprocess(X_k, Y_k)
    for _ in range(iterations):
        X_k, Y_k = generate_random_data(D, X_k, n)
        sdmd.update(X_k, Y_k)
        X_k = np.random.randn(n, 1)
    end_time = time.time()
    return (end_time - start_time)/iterations  # Return average time per iteration


def combined_plot_time_complexity(online_dmd_times, gradient_descent_times, streaming_dmd_times, n_values):
    """
    Plot the time complexity of Online DMD, Gradient Descent, and Streaming DMD update processes.
    
    Parameters:
    - online_dmd_times: List of times for Online DMD updates.
    - gradient_descent_times: List of times for Gradient Descent updates.
    - streaming_dmd_times: List of times for Streaming DMD updates.
    - n_values: List of n (dimension) values.
    """
    plt.figure(figsize=(6, 5), dpi=300)
    
    plt.scatter(n_values, online_dmd_times, color='red', alpha=0.5, label='Online DMD Update Time', marker='o')
    plt.scatter(n_values, gradient_descent_times, color='blue', alpha=0.5, label='Gradient Descent Update Time', marker='^')
    plt.scatter(n_values, streaming_dmd_times, color='green', alpha=0.5, label='Streaming DMD Update Time', marker='s')
    
    plt.xlabel('n (Dimension)', fontsize=14, fontweight='bold')
    plt.ylabel('Time (Seconds)', fontsize=14, fontweight='bold')
    plt.title('Time Complexity', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plt.show()


def save_time_complexity_data_to_desktop(n_values, online_dmd_times, gradient_descent_times, streaming_dmd_times, filename='time_complexity_data.csv'):
    """
    Save the time complexity data for various algorithms to a CSV file on the desktop.

    Parameters:
    - n_values: List of n (dimension) values tested.
    - online_dmd_times: List of times for Online DMD updates.
    - gradient_descent_times: List of times for Gradient Descent updates.
    - streaming_dmd_times: List of times for Streaming DMD updates.
    - filename: Name of the CSV file to save the data.
    """
    # Build the path to the desktop
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    full_path = os.path.join(desktop_path, filename)

    # Prepare the data for saving
    data = {
        'Dimension': n_values,
        'Online DMD Time': online_dmd_times,
        'Gradient Descent Time': gradient_descent_times,
        'Streaming DMD Time': streaming_dmd_times
    }

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(full_path, index=False)
    print(f"Results saved to {full_path}")



# Maximum allowed rank of the learned operator
max_rank = 50     
# Instantiate the DMD class
sdmd = StreamingDMD(max_rank=max_rank)
learning_rate = 0.04
iterations = 5
n_values = range(100, 1001, 100)  # Reduced range for quicker execution
online_dmd_times = []
gradient_descent_times = []
streaming_dmd_times = []  # To store time taken for each n


for n in n_values:
    print(n)
    D = np.random.rand(n, n)
    A_k_initial = np.zeros((n, n))
    X_k_initial = np.random.randn(n, 1)
    iterations = 10
    time_taken = run_online_dmd_update_timed(A_k_initial, D, X_k_initial, iterations)
    online_dmd_times.append(time_taken)

for n in n_values:
    print(n)
    D = np.random.rand(n, n)
    A_k = np.zeros((n, n))
    X_k_initial = np.random.randn(n, 1)
    time_taken = run_online_gradient_descent_timed(A_k, D, X_k_initial, learning_rate, iterations)
    gradient_descent_times.append(time_taken)


# Running the timed experiment for varying n
for n in n_values:
    print(n)
    D = np.random.rand(n, n)
    X_k_initial = np.random.randn(n, 1)
    time_taken = run_streaming_dmd_update_timed(max_rank=10, D=D, X_k_initial=X_k_initial, iterations=iterations)
    streaming_dmd_times.append(time_taken)


combined_plot_time_complexity(online_dmd_times, gradient_descent_times, streaming_dmd_times, n_values)

# After completing the time measurement for each algorithm, call the function with the appropriate data:
save_time_complexity_data_to_desktop(n_values, online_dmd_times, gradient_descent_times, streaming_dmd_times)


















