import numpy as np
from scipy.sparse import csr_matrix


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
    使用在线梯度下降进行单步更新，并应用循环矩阵约束
    """
    n, _ = X_k_new.shape
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T
    A_k -= learning_rate * grad
    return project_to_cyclic_matrix(A_k)


def projection_onto_upper_triangular(A):
    """
    将矩阵投影到上三角矩阵的空间
    """
    return np.triu(A)


def gradient_descent_update_triangular(A_k, X_k_new, Y_k_new, learning_rate):
    """
    使用在线梯度下降进行单步更新，并应用上三角矩阵约束
    """
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T 
    A_k -= learning_rate * grad
    A_k = projection_onto_upper_triangular(A_k)  # 应用上三角矩阵约束
    return A_k


def projection_onto_tridiagonal(A):
    """
    Efficiently project a matrix onto the space of tridiagonal matrices.
    """
    # Use NumPy's triu (upper triangle) and tril (lower triangle) functions
    # to keep only the main diagonal and the first diagonals above and below it.
    return np.triu(np.tril(A, 1), -1)


def gradient_descent_update_tridiagonal(A_k, X_k_new, Y_k_new,  learning_rate):
    """
    使用在线梯度下降进行单步更新，并应用三对角矩阵约束
    """
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T 
    A_k -= learning_rate * grad
    A_k = projection_onto_tridiagonal(A_k)  # 应用三对角矩阵约束
    return A_k


def projection_onto_symmetric(A):
    """
    将矩阵投影到对称矩阵的空间
    """
    return (A + A.T) / 2


def gradient_descent_update_symmetric(A_k, X_k_new, Y_k_new, learning_rate):
    """
    使用在线梯度下降进行单步更新，并应用对称矩阵约束
    """
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T 
    A_k -= learning_rate * grad
    A_k = projection_onto_symmetric(A_k)  # 应用对称矩阵约束
    return A_k


def make_sparse(A, threshold):
    """
    此函数接收一个密集矩阵A和一个阈值。
    它将A中绝对值小于阈值的元素设为零，有效地将矩阵稀疏化。
    然后返回一个CSR格式的稀疏矩阵。
    """
    A[np.abs(A) < threshold] = 0
    return csr_matrix(A)


def l1_soft_thresholding(A, lambda_):
    """
    此函数对矩阵A的每个元素应用软阈值操作。
    软阈值用于优化过程中的L1正则化（Lasso）。
    它通过lambda_将A的值缩小至零。
    """
    return np.sign(A) * np.maximum(np.abs(A) - lambda_, 0.0)


def gradient_descent_update_l1(A_k, X_k, Y_k, lambda_reg, learning_rate):
    """
    执行在线梯度下降的单步更新。
    这涉及计算损失函数相对于A_k的梯度，
    应用L1正则化，并通过按学习率缩放的负梯度方向更新A_k。
    它还包括了L1正则化的近端步骤。
    """
    # 计算损失函数的梯度
    grad = -2 * (Y_k - A_k @ X_k) @ X_k.T 
    # 按负梯度方向更新A_k
    A_k -= learning_rate * grad
    # 应用软阈值操作进行L1正则化
    A_k = l1_soft_thresholding(A_k, lambda_reg)
    return A_k


def proximal_unclear_norm(A, lambda_reg):
    """
    将矩阵投影到具有指定算子范数约束的矩阵空间
    """
    # 使用SVD分解矩阵A
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # 应用算子范数约束
    s = np.maximum(s - lambda_reg, 0)
    # 重建矩阵
    return U @ np.diag(s) @ Vt


def gradient_descent_update_unclear_norm(A_k, X_k_new, Y_k_new, lambda_reg, learning_rate):
    """
    使用在线梯度下降进行单步更新，并应用算子范数约束
    """
    # 计算梯度
    grad = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T
    # 梯度下降步骤
    A_k -= learning_rate * grad
    # 应用迹范数约束
    A_k = proximal_unclear_norm(A_k, lambda_reg)
    return A_k


def gradient_descent_update_l2(A_k, X_k_col, Y_k_col, lambda_val,learning_rate):
    # 计算梯度
    gradient = 2 * (A_k @ X_k_col - Y_k_col) @ X_k_col.T + 2 * lambda_val * A_k

    # 更新 A_k
    A_k -= learning_rate * gradient
    return A_k


def proximal_operator_norm(A, lambda_reg):
    """
    Project the matrix onto a space with a specified operator norm constraint,
    adjusting only the largest singular value.
    """
    # Perform SVD decomposition of matrix A
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # Apply the operator norm constraint only to the largest singular value
    s[0] = np.maximum(s[0] - lambda_reg, 0)  # Only adjust the largest singular value
    # s[0] = np.maximum(s[0] - lambda_reg, 1)  # Only adjust the largest singular value
    # s[-1] = np.minimum(s[-1] + lambda_reg, 1)# Adjust only the smallest singular value
    # Reconstruct the matrix
    return U @ np.diag(s) @ Vt


def gradient_descent_update_operator_norm(A_k, X_k_new, Y_k_new, lambda_reg, learning_rate):
    """
    使用在线梯度下降进行单步更新，并包含操作范数的次梯度
    """
    # 计算Frobenius范数的梯度
    grad_fro = -2 * (Y_k_new - A_k @ X_k_new) @ X_k_new.T
    A_k -= learning_rate * grad_fro 
    # 应用迹范数约束
    A_k = proximal_operator_norm(A_k, lambda_reg)
    return A_k




def proximal_operator_norm(A, lambda_reg):
    """
    Project the matrix onto a space with a specified operator norm constraint,
    adjusting only the largest singular value.
    """
    # Perform SVD decomposition of matrix A
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # Apply the operator norm constraint only to the largest singular value
    s[0] = np.maximum(s[0] - lambda_reg, 0)  # Only adjust the largest singular value
    # Reconstruct the matrix
    return U @ np.diag(s) @ Vt








