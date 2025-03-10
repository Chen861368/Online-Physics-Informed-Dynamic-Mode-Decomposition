
### Data and Code for Images and Algorithms in the Paper "Online Physics-Informed Dynamic Mode Decomposition: Theory and Applications"

If you're interested in learning more about my research, a brief introduction of **my personal research homepage** can be found on [GitHub](https://github.com/Chen861368/Personal-Research-Path).

---
#### Paper Abstract
Dynamic Mode Decomposition (DMD) has received increasing research attention due to its capability to analyze and model complex dynamical systems. However, it faces challenges in computational efficiency, noise sensitivity, and difficulty adhering to physical laws, which negatively affect its performance.
Addressing these issues, we present Online Physics-informed DMD (OPIDMD), a novel adaptation of DMD into a convex optimization framework. 
This approach not only ensures convergence to a unique global optimum, but also enhances the efficiency and accuracy of modeling dynamical systems in an online setting. Leveraging the Bayesian DMD framework, we propose a probabilistic interpretation of Physics-informed DMD (piDMD), examining the impact of physical constraints on the DMD linear operator. 

<p align="center">
  <img src="OPIDMD.png" alt="A schematic representation of the minimal realization time delay Koopman system identification algorithm process." width="60%" />
</p>

Further, we implement online proximal gradient descent and formulate specific algorithms to tackle problems with different physical constraints, enabling real-time solutions across various scenarios. Compared with existing algorithms such as Exact DMD, Online DMD, and piDMD, OPIDMD achieves the best prediction performance in short-term forecasting, e.g.  an R^2 value of 0.991 for noisy Lorenz system. The proposed method employs a time-varying linear operator, offering a promising solution for the real-time simulation and control of complex dynamical systems.

<p align="center">
  <img src="OPIDMD_algorithm.png" alt="A schematic representation of the minimal realization time delay Koopman system identification algorithm process." width="80%" />
</p>

#### Introduction
The paper is available on arXiv at: https://arxiv.org/abs/2412.03609. This document provides a detailed description of the data, images, and algorithms used in the paper "Online Physics-Informed Dynamic Mode Decomposition: Theory and Applications". The purpose is to ensure reproducibility and facilitate a deeper understanding of the methodologies employed.

The methods presented in the paper are tailored for time-series data characterized by a large number of samples but relatively low dimensionality in terms of the state vector (e.g., a state vector dimension of 30 and 1,000,000 data samples). Such data types are commonly encountered in sensor sampling systems, one of the most prevalent application scenarios. This repository retains the original code corresponding to each revision of the paper, stored in branches that reflect changes made during the writing and review process.


### Updates Based on Reviewer Comments
In response to the first round of review comments, we have added several new examples. The corresponding code for these examples is available in a new repository titled `Here’s-the-code-modified-according-to-the-reviewer’s-comments`, which can be accessed via the following link: [Here’s-the-code-modified-according-to-the-reviewer’s-comments](https://github.com/Chen861368/Online-Physics-Informed-Dynamic-Mode-Decomposition/tree/Here%E2%80%99s-the-code-modified-according-to-the-reviewer%E2%80%99s-comments). The filenames within the repository are descriptive and indicate the specific example they correspond to.


With the exception of the ‘Cylinder Flow’ example, the data for all other examples can be directly generated using the provided code, so there is no need to upload additional data. For the ‘Cylinder Flow’ example, the data can be downloaded independently from [https://www.databookuw.com/](https://www.databookuw.com/).

Below is an introduction to the files in the `main` directory.

#### Code Organization and Naming
The code provided in the repository is organized to follow the sequence of image generation as presented in the paper. The scripts are named according to the specific figures or sets of figures they generate, ensuring clarity and ease of use. Below is the structure of the codebase and the corresponding explanations:

1. **Main Scripts**:
    - `figure3,4,5(piDMD,OPIDMD).py`: Generates the data and plots for Figures 3, 4, and 5.
    - `figure6,7(OPIDMD).py`: Generates the data and plots for Figures 6 and 7 .
    - `figure10,11,12,13,14.py`: Generates the data and plots for Figures 10 to 14.

2. **Supplementary Material**:
    - `Supplementary Material Figure1(a).py`: Generates supplementary material for Figure 1(a).
    - `Supplementary Material Figure1(b).py`: Generates supplementary material for Figure 1(b).
    - `Supplementary Material Figure2(a).py`: Generates supplementary material for Figure 2(a).
    - `Supplementary Material Figure2(b).py`: Generates supplementary material for Figure 2(b).

3. **Utility Functions**:
    - `generate_data.py`: Contains functions for generating synthetic data used in the experiments.
    - `gradient_descent_functions.py`: Implements gradient descent algorithms used in the optimization process.
    - `visualization_functions.py`: Contains functions for visualizing data and generating plots.

4. **Modify Save Paths**: 
    - Before running the scripts, ensure that the save paths for the generated figures and data outputs are correctly set. This may involve modifying the paths in the scripts to match your local file system structure.
   

5. **Output**:
    - The script will process the data and generate the figures, saving them to the specified path. While the output figures will match those in the paper in terms of data and overall appearance, slight variations in formatting may occur due to differences in plotting libraries or versions.


#### Code Dependencies
This research integrates various algorithms, including OPIDMD, piDMD, and Streaming DMD, by referencing open-source libraries such as PyDMD and online DMD codebases. We are grateful for these open-source contributions, and we have acknowledged and cited them appropriately in the paper. These libraries provide robust implementations of dynamic mode decomposition methods, which significantly facilitated our comparative analysis and validation.

- **PyDMD**: A Python package for Dynamic Mode Decomposition, which offers a comprehensive set of tools for standard and extended DMD algorithms.
- **Online DMD**: Implementations of online variants of DMD, which are crucial for real-time analysis and updates.

To use these libraries, ensure they are installed in your environment. For example, you can install PyDMD using:
```bash
pip install pydmd
```

#### Conclusion
This document provides an overview of the data, images, and algorithms used in the paper "Online Physics-Informed Dynamic Mode Decomposition: Theory and Applications". By sharing these details, we aim to promote transparency, reproducibility, and further advancements in the field. We express our gratitude to the developers of the open-source libraries that made this research possible.


