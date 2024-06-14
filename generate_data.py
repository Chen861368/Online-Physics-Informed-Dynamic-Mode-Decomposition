# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse import random as sparse_random
from scipy.sparse.linalg import eigs


def generate_physical_dynamics_system(n):
    if n <= 0 or not isinstance(n, int):
        raise ValueError("The system dimension must be a positive integer.")

    # Initialize matrices with zeros
    M = np.zeros((n, n))
    C = np.zeros((n, n))
    K = np.zeros((n, n))

    # Random values for mass, damping, and stiffness
    m_values = np.random.uniform(1, 10, size=n)
    c_values = np.random.uniform(1, 5, size=n + 1)
    k_values = np.random.uniform(10, 50, size=n + 1)

    # Populate the Mass matrix M
    np.fill_diagonal(M, m_values)

    # Populate the Damping matrix C and Stiffness matrix K
    for i in range(n):
        C[i, i] = c_values[i] + c_values[i + 1]
        K[i, i] = k_values[i] + k_values[i + 1]
        if i < n - 1:
            C[i, i + 1] = C[i + 1, i] = -c_values[i + 1]
            K[i, i + 1] = K[i + 1, i] = -k_values[i + 1]

    return M, C, K

def create_physical_Rayleigh_damping_dynamics_system(n, a0, a1):
    if n <= 0 or not isinstance(n, int):
        raise ValueError("The system dimension must be a positive integer.")

    # Initialize matrices with zeros
    M = np.zeros((n, n))
    K = np.zeros((n, n))

    # Random values for mass, damping, and stiffness
    m_values = np.random.uniform(1, 10, size=n)
    k_values = np.random.uniform(10, 50, size=n + 1)

    # Populate the Mass matrix M
    np.fill_diagonal(M, m_values)

    # Populate the Damping matrix C and Stiffness matrix K
    for i in range(n):
        K[i, i] = k_values[i] + k_values[i + 1]
        if i < n - 1:
            K[i, i + 1] = K[i + 1, i] = -k_values[i + 1]
    # Create damping matrix C using Rayleigh damping
    C = a0 * M + a1 * K

    return M, C, K


def create_general_dynamics_equation(n):
    # 创建对角质量矩阵 M
    M = np.diag(np.random.rand(n))  # 假设质量在1到1之间

    # 创建对称刚度矩阵 K
    K = np.random.rand(n, n)  # 刚度在0到1之间
    K = (K + K.T) / 2  # 使其对称

    # 创建阻尼矩阵 C 
    C = np.random.rand(n, n)

    # 返回动力学方程的参数
    return M, C, K


def create_dynamics_equation(n, a0, a1):
    # 创建对角质量矩阵 M
    M = np.diag(np.random.rand(n))  # 假设质量在1到1之间

    # 创建对称刚度矩阵 K
    K = np.random.rand(n, n)  # 刚度在0到1之间
    K = (K + K.T) / 2  # 使其对称

    # 创建阻尼矩阵 C 使用瑞利阻尼
    C = a0 * M + a1 * K

    # 返回动力学方程的参数
    return M, C, K


def dynamics_system(n, M, C, K):
    """
    Creates the state-space representation of a dynamic system given the number of degrees of freedom (n),

    Returns:
    A (ndarray): System matrix
    B (ndarray): Input matrix
    s (ndarray): State matrix over time
    """

    # State-space matrices
    A = np.block([[np.zeros((n, n)), np.eye(n)],
                  [-np.linalg.inv(M) @ K, -np.linalg.inv(M) @ C]])
    B = np.block([[np.zeros((n, n))],
                  [np.linalg.inv(M)]])

    return A, B


def generate_random_data(D, X_k, n):
    """
    Generate new data X_k_new and corresponding Y_k_new using matrix D.
    
    Parameters:
    - D: The matrix used to generate Y_k_new.
    - n: The size of the vector X_k_new.
    
    Returns:
    - X_k_new: Newly generated data.
    - Y_k_new: Corresponding output data using matrix D.
    """
    Y_k = D @ X_k
    return X_k, Y_k

def generate_sequential_control_data(D, X_k, n):
    """
    Generate new data X_k_new and corresponding Y_k_new using matrices A and D, 
    where X_k_new = A @ X_k + np.random.randn(n, 1).
    
    Parameters:
    - A: The matrix used to transform X_k to X_k_new.
    - D: The matrix used to generate Y_k_new from X_k_new.
    - X_k: The current data vector.
    - n: The size of the vector X_k_new.
    
    Returns:
    - X_k_new: Newly generated data.
    - Y_k_new: Corresponding output data using matrix D.
    """
    # Generate the new data vector X_k_new
    u_k = np.random.randn(n, 1)
    X_k_new = D @ X_k + u_k
    return X_k, X_k_new, u_k

def generate_stable_sparse_matrix(n, density=0.3):
    """
    生成一个稳定的稀疏矩阵。

    参数:
    - n: 矩阵的大小 (n x n)。
    - density: 非零元素的密度。

    返回:
    - 稳定的稀疏矩阵 (CSR格式)。
    """
    # 随机生成稀疏矩阵
    D = sparse_random(n, n, density=density, format='csr', dtype=np.float64)
    
    # 计算最大的几个特征值
    eigenvalues, _ = eigs(D, k=1, which='LM')  # 计算最大的特征值
    
    max_eigenvalue_abs = np.abs(eigenvalues).max()
    
    # 如果最大特征值的绝对值大于1，通过该值缩放矩阵以确保所有特征值的绝对值小于等于1
    if max_eigenvalue_abs > 1:
        D /= max_eigenvalue_abs
    
    return D


# # 验证特征值确实都不大于1
# eigenvalues_10x10, _ = np.linalg.eig(D.toarray())
# print("特征值的最大绝对值:", np.abs(eigenvalues_10x10).max())

def generate_stable_matrix(n):
    # 步骤1: 生成一个随机的10x10矩阵
    A = np.random.rand(n, n)
    
    # 步骤2: 计算矩阵的特征值
    eigenvalues, _ = np.linalg.eig(A)
    
    # 步骤3: 获取最大特征值的绝对值
    max_eigenvalue_abs = np.abs(eigenvalues).max()
    
    # 如果最大特征值的绝对值大于1，通过该值缩放矩阵以确保所有特征值的绝对值小于等于1
    if max_eigenvalue_abs > 1:
        A /= max_eigenvalue_abs
    
    # 步骤4: 返回缩放后的矩阵
    return A


def generate_stable_triangular_matrix(n):
    # 步骤1: 生成一个随机的10x10矩阵
    A = np.random.rand(n, n)
    
    # 步骤2: 计算矩阵的特征值
    eigenvalues, _ = np.linalg.eig(A)
    
    # 步骤3: 获取最大特征值的绝对值
    max_eigenvalue_abs = np.abs(eigenvalues).max()
    
    # 如果最大特征值的绝对值大于1，通过该值缩放矩阵以确保所有特征值的绝对值小于等于1
    if max_eigenvalue_abs > 1:
        A /= max_eigenvalue_abs
    
    # 步骤4: 返回缩放后的矩阵
    return np.triu(A)












