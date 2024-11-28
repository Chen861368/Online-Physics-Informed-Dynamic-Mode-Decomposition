# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from visualization_functions import visualize_matrix,compute_and_plot_singular_values_scatter, plot_frobenius_diff
np.random.seed(0)
from sklearn.metrics import mean_squared_error, r2_score
from pydmd import PiDMD
from matplotlib.ticker import MaxNLocator
# Adjusted Newmark-beta method function
def newmark_beta_modified(dt, K, M, C, P, u0, v0, a0):
    n = K.shape[0]  # Determine the number of degrees of freedom
    nt = P.shape[1]  # Determine the total number of time steps
    gamma = 0.5
    beta = 0.25
    p = [1 / (beta * (dt ** 2)), 
         gamma / (beta * dt),
         1 / (beta * dt),
         0.5 / beta - 1,
         gamma / beta - 1,
         dt * (gamma / (2 * beta) - 1),
         dt * (1 - gamma),
         gamma * dt]
    
    # Storing the displacements, velocities, and accelerations
    u = np.zeros((n, nt))
    v = np.zeros((n, nt))
    a = np.zeros((n, nt))

    # Set initial conditions
    u[:, 0] = u0
    v[:, 0] = v0
    a[:, 0] = a0

    # Equivalent stiffness matrix
    K_ = K + p[0] * M + p[1] * C
    K_inv = np.linalg.inv(K_)

    # Time stepping solution
    for i in range(1, nt):
        # Apply external force only at the first time step
        if i == 1:
            P_ = P[:, 0] + M.dot(p[0] * u[:, i-1] + p[2] * v[:, i-1] + p[3] * a[:, i-1]) + C.dot(p[1] * u[:, i-1] + p[4] * v[:, i-1] + p[5] * a[:, i-1])
        else:
            P_ = M.dot(p[0] * u[:, i-1] + p[2] * v[:, i-1] + p[3] * a[:, i-1]) + C.dot(p[1] * u[:, i-1] + p[4] * v[:, i-1] + p[5] * a[:, i-1])
            
        u[:, i] = K_inv.dot(P_)
        a[:, i] = p[0] * (u[:, i] - u[:, i-1]) - p[2] * v[:, i-1] - p[3] * a[:, i-1]
        v[:, i] = v[:, i-1] + p[6] * a[:, i-1] + p[7] * a[:, i]
        
    return u, v, a


def plot_displacements(time, displacements, save_path="C:\\Users\\HIT\\Desktop"):
    """
    Plots the displacement of each mass over time with stylistic adjustments
    for enhanced readability and aesthetics, and saves the plot as a PDF.
    """
    sns.set(style="white", context="talk")  # Use a white background and context suitable for presentations

    plt.figure(figsize=(12, 8), dpi=300)  # Set figure size and DPI
    colors = sns.color_palette("muted", displacements.shape[0])  # Use a muted color palette

    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16

    for i, color in zip(range(displacements.shape[0]), colors):
        plt.plot(time, displacements[i, :], label=f'Mass {i+1}', color=color)

    # Customize labels with explicit font sizes
    plt.xlabel('Time (seconds)', fontsize=25)
    plt.ylabel('Displacement', fontsize=25)

    # Enlarge legend line and text
    plt.legend(fontsize='x-large', handlelength=2)

    # Explicitly set tick label sizes
    plt.tick_params(axis='both', which='major', labelsize=20)

    # Use dashed grid lines
    plt.grid(True, linestyle='--')

    plt.tight_layout()
    plt.savefig(f"{save_path}/displacements_visualization.pdf", format='pdf', bbox_inches='tight')
    plt.show()


def plot_frequency_spectrum(time, displacements, save_path="C:\\Users\\HIT\\Desktop"):
    """
    Plots the frequency spectrum of each mass's displacement using FFT,
    with stylistic adjustments for enhanced readability and aesthetics.
    """
    sns.set(style="white", context="talk")  # Use a white background and context suitable for presentations

    plt.figure(figsize=(12, 8), dpi=300)  # Set figure size and DPI

    # Calculate FFT and frequencies
    dt = time[1] - time[0]  # Calculate timestep
    n = displacements.shape[1]
    freq = np.fft.fftfreq(n, d=dt)[:n//2]  # Only positive frequencies

    colors = sns.color_palette("muted", displacements.shape[0])  # Use a muted color palette

    # Set global font to Times New Roman and increase size
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16

    for i, color in zip(range(displacements.shape[0]), colors):
        fft_vals = np.fft.fft(displacements[i, :])
        fft_theo = 2.0/n * np.abs(fft_vals[:n//2])  # Single-sided spectrum

        plt.plot(freq, fft_theo, label=f'Mass {i+1}', color=color)

    # Customize labels with explicit font sizes
    plt.xlabel('Frequency (Hz)', fontsize=25)
    plt.ylabel('Amplitude', fontsize=25)

    plt.xlim(0, 0.4)  # Adjust frequency display range as needed

    # Enlarge legend line and text
    plt.legend(fontsize='x-large', handlelength=2)

    # Explicitly set tick label sizes
    plt.tick_params(axis='both', which='major', labelsize=20)

    # Use dashed grid lines
    plt.grid(True, linestyle='--')

    plt.tight_layout()
    plt.savefig(f"{save_path}\\frequency_visualization.pdf", format='pdf', bbox_inches='tight')
    plt.show()
    
    

def save_plot_data_and_spectrum_to_desktop(time, displacements, accelerations, filename_displacements='displacements_data.csv', filename_spectrum='frequency_spectrum_data.csv'):
    """
    Save the displacement data and frequency spectrum data to CSV files on the desktop.

    Parameters:
    - time: Array of time values.
    - displacements: Array of displacement values for each mass.
    - accelerations: Array of acceleration values for each mass.
    - filename_displacements: Name of the CSV file to save the displacement data.
    - filename_spectrum: Name of the CSV file to save the frequency spectrum data.
    """
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    
    # Save displacements data
    displacements_data = pd.DataFrame(displacements.T, index=time, columns=[f'Mass {i+1}' for i in range(displacements.shape[0])])
    displacements_data.index.name = 'Time'
    displacements_data.to_csv(os.path.join(desktop_path, filename_displacements))
    
    # Calculate frequency spectrum data
    dt = time[1] - time[0]
    n = len(time)
    freq = np.fft.fftfreq(n, d=dt)[:n//2]
    spectrum_data = pd.DataFrame(index=freq[freq <= 0.5])  # Frequency limit to 50Hz

    for i in range(accelerations.shape[0]):
        fft_vals = np.fft.fft(accelerations[i, :])
        fft_theo = 2.0/n * np.abs(fft_vals[:n//2])
        spectrum_data[f'Mass {i+1}'] = fft_theo[freq <= 0.5]

    spectrum_data.index.name = 'Frequency'
    spectrum_data.to_csv(os.path.join(desktop_path, filename_spectrum))

    print(f"Displacement data saved to {os.path.join(desktop_path, filename_displacements)}")
    print(f"Frequency spectrum data saved to {os.path.join(desktop_path, filename_spectrum)}")


def compute_dmd_matrix(displacements, rank=None):
    """
    Compute the Dynamic Mode Decomposition (DMD) matrix A with an option to use truncated SVD.

    Parameters:
    - displacements: numpy.ndarray, a matrix containing the displacements of each mass at each time step.
    - rank: int, optional, the rank for the truncated SVD.

    Returns:
    - A: numpy.ndarray, the approximated system matrix describing the dynamics.
    """
    # Split the displacement data into X and Y matrices
    X = displacements[:, :-1]
    Y = displacements[:, 1:]

    # Perform the Singular Value Decomposition (SVD) of X
    U, Sigma, Vh = np.linalg.svd(X, full_matrices=False)

    # If a rank is specified, truncate the SVD results
    if rank is not None and rank > 0:
        U = U[:, :rank]
        Sigma = Sigma[:rank]
        Vh = Vh[:rank, :]

    # Construct the diagonal matrix for the pseudo-inverse
    Sigma_inv = np.linalg.inv(np.diag(Sigma))

    # Compute the DMD matrix A
    A = Y @ Vh.T @ Sigma_inv @ U.T

    return A

def plot_reconstructed_vs_actual(time, actual_data, reconstructed_data, mass_index, name,save_path="C:\\Users\\HIT\\Desktop"):
    """
    Plots actual vs reconstructed data for a selected mass (dimension) with stylistic adjustments
    for enhanced readability and aesthetics, and saves the plot as a PDF.

    Parameters:
    - time: numpy.ndarray, the time steps.
    - actual_data: numpy.ndarray, the actual data matrix for a single dimension.
    - reconstructed_data: numpy.ndarray, the data matrix reconstructed by DMD for a single dimension.
    - mass_index: int, index of the mass (dimension) to plot.
    - save_path: str, the directory path where the plot will be saved.
    """
    sns.set(style="white", context="talk")  # Use a white background and context suitable for presentations

    plt.figure(figsize=(12, 8), dpi=300)  # Set figure size and DPI

    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16

    # Plot actual data with a solid black line
    plt.plot(time, actual_data, label=f'Noisy data', linestyle='-', marker='', color='blue', linewidth=2.5)

    # Plot reconstructed data with a dashed red line
    plt.plot(time, reconstructed_data, label=f'Actual data', linestyle=':', marker='', color='orange', linewidth=2.5)

    # Customize labels with explicit font sizes
    plt.xlabel('Time (seconds)', fontsize=25)
    plt.ylabel(name, fontsize=25)

    # Enlarge legend line and text, add a black edge to the legend with a white background for visibility
    legend = plt.legend(fontsize='x-large', handlelength=2, edgecolor='black', frameon=True, fancybox=False)
    # Set the linewidth of the legend border
    legend.get_frame().set_linewidth(1.5)

    # Explicitly set tick label sizes
    plt.tick_params(axis='both', which='major', labelsize=20)

    # Use dashed grid lines for better readability
    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    # Generate the filename based on the mass index
    filename = f"{save_path}/noised_displacement_vs_actual_mass_{mass_index+1}.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()




def visualize_eigenvalues(A, A_dmd, A_pidmd, A_low, A_cirr, A_diag, save_path="C:\\Users\\HIT\\Desktop"):
    """
    Visualize the eigenvalues of matrices on the complex plane with a square axis frame
    and centered around the eigenvalues for better visibility. The legend has a black edge
    and the grid lines are dashed.
    """
    eigs = np.linalg.eigvals(A)
    eigs_dmd = np.linalg.eigvals(A_dmd)
    eigs_pidmd = np.linalg.eigvals(A_pidmd)
    eigs_tri = np.linalg.eigvals(A_low)
    eigs_circ = np.linalg.eigvals(A_cirr)
    eigs_diag = np.linalg.eigvals(A_diag)
    
    # Set the font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 17

    fig, ax = plt.subplots(figsize=(20, 6), dpi=300)

    # Find the maximum range for the real and imaginary parts
    all_eigs = np.concatenate((eigs, eigs_dmd, eigs_pidmd))
    max_range = max(np.ptp(all_eigs.real), np.ptp(all_eigs.imag))

    # Set the limits to center the eigenvalues within a square plot
    real_mid = (max(all_eigs.real) + min(all_eigs.real)) / 2
    imag_mid = (max(all_eigs.imag) + min(all_eigs.imag)) / 2
    ax.set_xlim(real_mid - max_range / 1.5, real_mid + max_range / 1.5)
    ax.set_ylim(imag_mid - max_range / 1.7, imag_mid + max_range / 1.7)

    # Plot eigenvalues
    ax.scatter(eigs.real, eigs.imag, c="black", marker='o', label='Truth')
    ax.scatter(eigs_dmd.real, eigs_dmd.imag, c="red", marker='^', label='DMD')
    ax.scatter(eigs_tri.real, eigs_tri.imag, c="green", marker='s', label='Lower triangular')
    ax.scatter(eigs_circ.real, eigs_circ.imag, c="purple", marker='*', label='Circulant')
    ax.scatter(eigs_diag.real, eigs_diag.imag, c="orange", marker='d', edgecolors='orange', label='Diagonal', alpha=0.7)
    ax.scatter(eigs_pidmd.real, eigs_pidmd.imag, c="blue", marker='x', label='OPIDMD')
    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")

    # Grid with dashed lines
    ax.grid(True, linestyle='--')

    # Add legend with black edge and set the linewidth of the legend's frame
    legend = ax.legend(loc='upper right', edgecolor='black', frameon=True, fancybox=False)
    legend.get_frame().set_linewidth(1.5)
    
    # Set aspect of the plot to be equal
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f"{save_path}\\eigenvalues_visualization.pdf", bbox_inches='tight')
    plt.show()


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



def gradient_descent_update_circulant(A_k, x_k, y_k, learning_rate):
    """
    Perform a gradient descent update step for the circulant matrix A_k using a single column x_k and corresponding y_k.
    
    Parameters:
    - A_k: Current matrix to be updated.
    - x_k: Input vector used in the update step.
    - y_k: Target vector used in the update step.
    - learning_rate: The learning rate for gradient descent.
    
    Returns:
    - A_k: Updated circulant matrix after the gradient descent step.
    """
    # Calculate the gradient
    grad = -2 * (y_k - A_k.dot(x_k)) * x_k.T
    # Update A_k
    A_k -= learning_rate * grad
    # Ensure A_k remains a circulant matrix
    A_k = np.tril(A_k)
    return project_to_cyclic_matrix(A_k)





def evaluate_predictions(actual, predicted):
    """
    Evaluate the prediction quality of a model using MSE and R^2 metrics.

    Parameters:
    - actual (np.ndarray): The actual data.
    - predicted (np.ndarray): The predicted data.

    Returns:
    - mse (float): Mean Squared Error of the predictions.
    - r2 (float): Coefficient of Determination (R^2) of the predictions.
    """
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    return mse, r2


class OnlineDMD:

    def __init__(self, n: int, weighting: float = 0.9) -> None:
        """Creat an object for online DMD
        Usage: odmd = OnlineDMD(n, weighting)

        Args:
            n (int): state dimension x(t) as in  z(t) = f(z(t-1)) or y(t) = f(t, x(t))
            weighting (float, optional): exponential weighting factor
                smaller value allows more adpative learning, 
                but too small weighting may result in model identification instability (relies only on limited recent snapshots). 
                Defaults to 0.9.
        """
        assert isinstance(n, int) and n >= 1
        weighting = float(weighting)
        assert weighting > 0 and weighting <= 1

        self.n = n
        self.weighting = weighting
        self.timestep = 0
        self.A = np.zeros([n, n])
        self._P = np.zeros([n, n])
        # initialize
        self._initialize()
        self.ready = False

    def _initialize(self) -> None:
        """Initialize online DMD with epsilon small (1e-15) ghost snapshot pairs before t=0"""
        epsilon = 1e-15
        alpha = 1.0 / epsilon
        self.A = np.random.randn(self.n, self.n)
        self._P = alpha * np.identity(self.n)  # inverse of cov(X)


    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Update the DMD computation with a new pair of snapshots (x,y)
        Here, if the (discrete-time) dynamics are given by z(t) = f(z(t-1)),
        then (x,y) should be measurements correponding to consecutive states
        z(t-1) and z(t).
        Usage: odmd.update(x, y)

        Args:
            x (np.ndarray): 1D array, shape (n, ), x(t) as in y(t) = f(t, x(t))
            y (np.ndarray): 1D array, shape (n, ), y(t) as in y(t) = f(t, x(t))
        """
        assert x is not None and y is not None
        x, y = np.array(x), np.array(y)
        assert np.array(x).shape == np.array(y).shape
        assert np.array(x).shape[0] == self.n

        # compute P*x matrix vector product beforehand
        Px = self._P.dot(x)
        # compute gamma
        gamma = 1.0 / (1 + x.T.dot(Px))
        # update A
        self.A += np.outer(gamma * (y - self.A.dot(x)), Px)
        # update P, group Px*Px' to ensure positive definite
        self._P = (self._P - gamma * np.outer(Px, Px)) / self.weighting
        # ensure P is SPD by taking its symmetric part
        self._P = (self._P + self._P.T) / 2

        # time step + 1
        self.timestep += 1

        if self.timestep >= 2 * self.n:
            self.ready = True
        return self.A



def run_online_dmd_update(combined_matrix_noise, D, iterations):
    """
    Run the online DMD update process.
    
    Parameters:
    - combined_matrix_noise: Matrix containing noise, where each column represents a time step.
    - D: The target matrix to compare against.
    - iterations: Number of iterations to run the DMD update.
    
    Returns:
    - odmd.A: Updated DMD matrix after all iterations.
    - frobenius_diffs: List of Frobenius norm differences between the updated DMD matrix and D at each iteration.
    """
    n = combined_matrix_noise.shape[0]  # Ensure n matches the number of rows in combined_matrix_noise
    odmd = OnlineDMD(n, 1)  # Initialize OnlineDMD with the correct value of n
    frobenius_diffs = []

    for idx in range(iterations):
        # Extract x_k and flatten it
        x_k = combined_matrix_noise[:, idx:idx+1].flatten()
        # Extract y_k and flatten it
        y_k = combined_matrix_noise[:, idx+1:idx+2].flatten()

        # Update DMD matrix using flattened x_k and y_k
        A_k = odmd.update(x_k, y_k)
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)

    return odmd.A, frobenius_diffs



def run_online_gradient_descent_circulant(combined_matrix_noise, learning_rate, D, iterations):
    """
    Run online gradient descent using combined_matrix_noise data to update the circulant matrix A_k column by column.
    
    Parameters:
    - combined_matrix_noise: Matrix containing noise data, where each column represents a time step.
    - learning_rate: The learning rate for gradient descent.
    - D: The target matrix to compare against.
    - iterations: Number of iterations to run the gradient descent.
    
    Returns:
    - A_k: Updated circulant matrix after gradient descent.
    - frobenius_diffs: List of Frobenius norm differences between the updated A_k and D at each iteration.
    """
    n = combined_matrix_noise.shape[0]
    A_k = np.zeros((n, n))  # Initialize A_k as an all-zero matrix
    frobenius_diffs = []

    # Use the number of columns in combined_matrix_noise as the number of iterations
    for idx in range(iterations):
        x_k = combined_matrix_noise[:, idx:idx+1]  # Extract one column as x_k
        y_k = combined_matrix_noise[:, idx+1:idx+2]  # Extract the next column as y_k
        A_k = gradient_descent_update_circulant(A_k, x_k, y_k, learning_rate)  # Update A_k
        frobenius_diff = np.linalg.norm(A_k - D, 'fro')
        frobenius_diffs.append(frobenius_diff)

    return A_k, frobenius_diffs


def add_nonstationary_gaussian_noise(signal, noise_ratio):
    """
    Add non-stationary Gaussian noise to a signal. The noise added to each sample is proportional
    to the magnitude of the signal at that point.

    Parameters:
    - signal (np.ndarray): The original signal.
    - noise_ratio (float): The ratio of the noise amplitude to the signal amplitude.

    Returns:
    - noisy_signal (np.ndarray): Signal with added non-stationary Gaussian noise.
    """
    # Calculate noise standard deviation for each sample
    noise_std_per_sample = np.abs(signal) * noise_ratio

    # Generate non-stationary Gaussian noise
    noise = noise_std_per_sample * np.random.normal(0, 1, signal.shape)

    # Add noise to the original signal
    noisy_signal = signal + noise
    return noisy_signal


def plot_comparison(time, actual_data, piDMD_data, DMD_data, mass_index, name,save_path="C:\\Users\\HIT\\Desktop"):
    """
    Plot a comparison of actual data with predictions from online piDMD and online DMD, using styles akin to Nature or Science publications.
    
    Parameters:
    - time: Array of time steps.
    - actual_data: Actual data array.
    - piDMD_data: Predicted data array from online piDMD.
    - DMD_data: Predicted data array from online DMD.
    - mass_index: Index of the mass for labeling.
    - save_path: Path to save the plot as a PDF.
    """
    # Set style
    sns.set(style="white", context="talk")
    plt.figure(figsize=(12, 8), dpi=300)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    # Plotting with updated colors and line styles
    plt.plot(time, actual_data, label='Actual Data', color='grey', linewidth=2)
    plt.plot(time, piDMD_data, label='OPIDMD', color='blue', linestyle='-.', linewidth=2)
    plt.plot(time, DMD_data, label='Online DMD', color='orange', linestyle=':', linewidth=2.5)

    # Labels and title with updated font sizes
    plt.xlabel('Time (seconds)', fontsize=25)
    plt.ylabel(name, fontsize=25)
    # plt.ylabel('Velocity', fontsize=25)

    # Legend
    legend = plt.legend(fontsize='x-large', loc='upper right', edgecolor='black', frameon=True, fancybox=False)
    legend.get_frame().set_linewidth(1.5)

    # Grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Saving
    filename = f"{save_path}/comparison_Displacement_mass_{mass_index+1}.pdf"
    # filename = f"{save_path}/comparison_velocity_mass_{mass_index+1}.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()  # Display the figure

def evaluate_predictions(actual, predicted):
    """
    Evaluate the prediction quality of a model using MSE and R^2 metrics.
    Adjusts for complex data by taking the real part.

    Parameters:
    - actual (np.ndarray): The actual data, which can be complex.
    - predicted (np.ndarray): The predicted data, which can be complex.

    Returns:
    - mse (float): Mean Squared Error of the predictions.
    - r2 (float): Coefficient of Determination (R^2) of the predictions.
    """
    actual_real = np.real(actual)
    predicted_real = np.real(predicted)

    mse = mean_squared_error(actual_real, predicted_real)
    r2 = r2_score(actual_real, predicted_real)
    
    return mse, r2



# Constants and settings
m = 1  # Mass
c = 0.06 # Damping coefficient
k = 1  # Stiffness coefficient
n = 5  # Number of degrees of freedom
t_end = 1000  # End time
dt = 0.005  # Time step
time = np.arange(0, t_end + dt, dt)
noise_ratio = 0.25
rank=10
learning_rate = 40000

# Mass matrix M
M = m * np.eye(n)

# Stiffness matrix K
K = np.diag([2*k] * (n - 1) + [k]) - np.diag([k] * (n-1), 1) - np.diag([k] * (n-1), -1)

# Damping matrix C
C = (c/k) * K

# External force F(t) applied only at the initial time step to the last mass
F_initial = np.zeros((n, len(time)))
F_initial[-1, 0] = np.random.normal(0, 0.5)  # Apply random force only at the first time step

# Initial conditions
x0 = np.zeros(n)
v0 = np.zeros(n)
a0 = np.linalg.inv(M).dot(F_initial[:, 0] - C.dot(v0) - K.dot(x0))


# Call the modified Newmark-beta method
displacements, velocities, accelerations = newmark_beta_modified(dt, K, M, C, F_initial, x0, v0, a0)

noisy_displacements = add_nonstationary_gaussian_noise(displacements, noise_ratio)
noisy_velocities = add_nonstationary_gaussian_noise(velocities, noise_ratio)
combined_matrix = np.vstack((displacements, velocities))
combined_matrix_noise = np.vstack((noisy_displacements, noisy_velocities))


A = compute_dmd_matrix(combined_matrix,rank)
print(combined_matrix.shape)
# Matrices obtained from OPTDMD and online DMD using the first 800 data points
A_k_optimized, frobenius_diffs = run_online_gradient_descent_circulant(combined_matrix_noise, learning_rate, A, 800)
online_dmd, frobenius_diffs2 = run_online_dmd_update(combined_matrix_noise, A, 800)

# Matrices obtained from OPTDMD and online DMD using all data points
A_k_optimized_full, frobenius_diffs_full = run_online_gradient_descent_circulant(combined_matrix_noise, learning_rate, A, combined_matrix_noise.shape[1] - 1)
online_dmd_full, frobenius_diffs2_full = run_online_dmd_update(combined_matrix_noise, A, combined_matrix_noise.shape[1] - 1)

# Matrices obtained from piDMD using the first 800 data points
triangular_matrix_pi = PiDMD(manifold="lowertriangular", compute_A=True).fit(combined_matrix_noise[:, 0:800])
circulant_matrix_pi = PiDMD(manifold="circulant", compute_A=True).fit(combined_matrix_noise[:, 0:800])
diagonal_matrix_pi = PiDMD(manifold="diagonal", compute_A=True).fit(combined_matrix_noise[:, 0:800])




X = combined_matrix[:, :-1]

reconstructed_data1 = A_k_optimized @ X
reconstructed_data2 = online_dmd @ X
reconstructed_data3 = triangular_matrix_pi.A @ X
reconstructed_data4 = circulant_matrix_pi.A @ X
reconstructed_data5 = diagonal_matrix_pi.A @ X



# Assuming reconstructed_data2 and combined_matrix are defined
mse_pi, r2_pi = evaluate_predictions(combined_matrix[:,1:], reconstructed_data1)
mse_dmd, r2_dmd = evaluate_predictions(combined_matrix[:,1:], reconstructed_data2)

print(f"Online piDMD Mean Squared Error (MSE): {mse_pi}")
print(f"Online DMD Mean Squared Error (MSE): {mse_dmd}")

print(f"Online piDMD Coefficient of Determination (R^2): {r2_pi}")
print(f"Online DMD Coefficient of Determination (R^2): {r2_dmd}")

    
# Example usage
mse_dmd_circulant_matrix, r2_dmd1 = evaluate_predictions(combined_matrix[:,1:], reconstructed_data3)
mse_dmd_triangular_matrix, r2_dmd2 = evaluate_predictions(combined_matrix[:,1:], reconstructed_data4)
mse_dmd_diagonal_matrix, r2_dmd3 = evaluate_predictions(combined_matrix[:,1:], reconstructed_data5)

print(f"circulant_matrix Coefficient of Determination (R^2): {r2_dmd1:.3f}")
print(f"triangular_matrix Coefficient of Determination (R^2): {r2_dmd2:.3f}")
print(f"diagonal_matrix Coefficient of Determination (R^2): {r2_dmd3:.3f}")



# Plot the displacement and acceleration frequency spectra
plot_displacements(time, displacements)
plot_frequency_spectrum(time, accelerations)

# Comparison of noisy and noiseless displacements
for i in range(5):  # Assuming 5 masses
    plot_reconstructed_vs_actual(time[:], noisy_displacements[i, :], displacements[i, :], i, "Displacement")

# Comparison of noisy and noiseless velocities
for i in range(5):  # Assuming 5 masses
    plot_reconstructed_vs_actual(time[:], noisy_velocities[i, :], velocities[i, :], i, "Velocity")

# Results of the Original DMD Figure 12(a)
visualize_matrix(A, "A")

# Results of OPTDMD with 800 data points Figure 12(b)
visualize_matrix(A_k_optimized, "A_k_optimized")

# Results of Online DMD with 800 data points Figure 12(c)
visualize_matrix(online_dmd, "online_dmd")

# Results of Eigenvalue Spectrum Figure 12(d)
visualize_eigenvalues(A, online_dmd, A_k_optimized, triangular_matrix_pi.A, circulant_matrix_pi.A, diagonal_matrix_pi.A)

# Results of OPTDMD with all data points Figure 12(e)
visualize_matrix(A_k_optimized_full, "A_k_optimized_full")

# Results of Online DMD with all data points Figure 12(f)
visualize_matrix(online_dmd_full, "online_dmd_full")

# Matrices obtained from piDMD using the first 800 data points Figure 13
visualize_matrix(circulant_matrix_pi.A, "circulant_matrix")
visualize_matrix(triangular_matrix_pi.A, "triangular_matrix")
visualize_matrix(diagonal_matrix_pi.A, "diagonal_matrix")

# Comparison of OPTDMD and Online DMD Figure 14
for i in range(10):
    if i < 5:
        plot_comparison(time[:-1], combined_matrix[i, 1:], reconstructed_data1[i, :], reconstructed_data2[i, :], i, "Displacement")
    else:
        plot_comparison(time[:-1], combined_matrix[i, 1:], reconstructed_data1[i, :], reconstructed_data2[i, :], i, "Velocity")

































