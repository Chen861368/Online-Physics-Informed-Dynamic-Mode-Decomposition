# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def visualize_matrix(A, title,save_path="C:\\Users\\HIT\\Desktop"):
    vmax = np.abs(A).max()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    X, Y = np.meshgrid(np.arange(A.shape[1]+1), np.arange(A.shape[0]+1))
    ax.invert_yaxis()
    pos = ax.pcolormesh(X, Y, A.real, cmap="seismic", vmax=vmax, vmin=-vmax)
    plt.title(f"{title}")

    cbar = fig.colorbar(pos, ax=ax)
    cbar.ax.tick_params(labelsize=10)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()
    plt.savefig(f"{save_path}/matrix_visualization.pdf", format='pdf', bbox_inches='tight')

def visualize_eigenvalues(A, save_path="C:\\Users\\HIT\\Desktop"):
    eigs, _ = np.linalg.eig(A)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.axvline(x=0, c="k", lw=1)
    ax.axhline(y=0, c="k", lw=1)
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(t), np.sin(t), c="gray", ls="--")  # Unit circle
    ax.scatter(eigs.real, eigs.imag, c="blue", alpha=0.7)  # Eigenvalues
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.axis("equal")
    ax.set_aspect('equal', 'box')
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()
    # plt.savefig(f"{save_path}/eigenvalues_visualization.pdf", format='pdf', bbox_inches='tight')


def visualize_matrix_and_eigenvalues(A, save_path):
    # 计算用于两个图的共同参数
    vmax = np.abs(A).max()
    eigs, _ = np.linalg.eig(A)
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.size'] = 15

    # 创建一个图和两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # 第一个子图：矩阵可视化
    X, Y = np.meshgrid(np.arange(A.shape[1]+1), np.arange(A.shape[0]+1))
    ax1.invert_yaxis()
    pos = ax1.pcolormesh(X, Y, A.real, cmap="seismic", vmax=vmax, vmin=-vmax)
    fig.colorbar(pos, ax=ax1)
    ax1.set_title("Matrix Visualization")

    # 第二个子图：特征值可视化
    ax2.axvline(x=0, c="k", lw=1)
    ax2.axhline(y=0, c="k", lw=1)
    t = np.linspace(0, 2 * np.pi, 100)
    ax2.plot(np.cos(t), np.sin(t), c="gray", ls="--")  # 单位圆
    ax2.scatter(eigs.real, eigs.imag, c="blue", alpha=0.7)  # 特征值
    ax2.set_title("Eigenvalues in the Complex Plane")
    ax2.set_xlabel("Real")
    ax2.set_ylabel("Imag")
    ax2.axis("equal")

    plt.tight_layout()
    # 保存为PDF格式到指定路径
    plt.savefig(f"{save_path}/matrix_and_eigenvalues_visualization.pdf", format='pdf')





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
    plt.figure(figsize=(8, 6), dpi=300)

    # Scatter plot with color and marker adjustments for elegance and visibility
    plt.scatter(lambda_values, eigenvalue_counts, color='blue', alpha=0.75, edgecolors='w', linewidth=0.5, s=50)

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




def plot_frobenius_diff(frobenius_diffs):
    """
    Plot the Frobenius norm difference vs. iteration in a publication-quality style.
    """
    fig, ax = plt.subplots(dpi=300)
    ax.plot(frobenius_diffs, color='blue', lw=2, label='Frobenius Norm Difference')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Frobenius Norm Difference')
    ax.set_title('Frobenius Norm Difference vs. Iteration')
    ax.legend()
    plt.tight_layout()
    plt.show()


def compute_and_plot_singular_values_scatter(matrix):
    """
    计算给定矩阵的奇异值，并通过散点图以科研风格展示。

    参数:
    - matrix: 需要计算奇异值的矩阵，一个二维numpy数组。
    """
    # 计算奇异值
    U, singular_values, Vt = np.linalg.svd(matrix)
    
    # 设定绘图样式为白色背景，类似于科研期刊风格
    plt.style.use('seaborn-whitegrid')
    
    # 使用散点图展示奇异值
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(range(len(singular_values)), singular_values, color='blue', edgecolor='black', s=50)
    plt.xlabel('Singular Value Index', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.title('Singular Values of the Matrix', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()


def plot_gradient_descent_paths(all_frobenius_diffs, dpi=300):
    """
    Plot the paths of Frobenius norm differences for multiple gradient descent runs in a style
    similar to that seen in publications such as Nature or Science.

    Parameters:
    - all_frobenius_diffs: A list of lists containing Frobenius norm differences for each run.
    - dpi: Dots per inch for the figure's resolution.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(8, 6), dpi=dpi)
        # Set the color palette to a professional and visually appealing one
        palette = plt.get_cmap('tab10')

        # Plot each run's path with a more transparent and thinner line
        for idx, diff in enumerate(all_frobenius_diffs):
            plt.plot(diff, color=palette(idx % 10), alpha=0.3, linewidth=1)  # Each experiment's path

        # Calculate and plot the average path with a thicker line and distinct color
        mean_diff = np.mean(all_frobenius_diffs, axis=0)
        plt.plot(mean_diff, color='blue', linewidth=2, label='Average Path')  # Average path

        # Customize the scale, title, labels, and legend for clarity and visual appeal
        plt.yscale('log')
        plt.title('Gradient Descent Convergence Over Multiple Runs', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration (k)', fontsize=12)
        plt.ylabel('Normalized Frobenius Norm Difference', fontsize=12)
        plt.legend(fontsize=10, loc='upper right')

        # Customize grid lines for clarity
        plt.grid(True, which="both", linestyle='--', linewidth=0.5)

        # Set the axes' display range to encompass all data
        plt.xlim([0, len(mean_diff) - 1])
        plt.ylim([np.min(all_frobenius_diffs), np.max(all_frobenius_diffs)])

        # Remove top and right border lines for a cleaner look
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set font size for tick parameters for readability
        plt.tick_params(axis='both', which='major', labelsize=10)
        
        # Ensure the layout fits well without cutting off elements
        plt.tight_layout()
        plt.show()

# 注意：这段代码已经进行了修改，以尝试提升图表的美观度和专业性，接近Nature或Science的风格。
# 请在合适的Python环境中执行它，因为这里无法直接运行matplotlib代码。
