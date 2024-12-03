
### Data and Code for Images and Algorithms in the Paper "Online Physics-Informed Dynamic Mode Decomposition: Theory and Applications"

#### Introduction
This document provides a detailed description of the data, images, and algorithms used in the paper "Online Physics-Informed Dynamic Mode Decomposition: Theory and Applications". The purpose is to ensure reproducibility and facilitate a deeper understanding of the methodologies employed.

The methods presented in the paper are designed for time-series data that feature a large number of samples but relatively low dimensionality (e.g., 30 × 1,000,000). Such data types are commonly encountered in sensor sampling systems, one of the most prevalent application scenarios. This repository retains the original code corresponding to each revision of the paper, stored in branches that reflect changes made during the writing and review process.

### Updates Based on Reviewer Comments
In response to the first round of review comments, we have added several new examples. The corresponding code for these examples has been placed in a new repository titled `Here’s-the-code-modified-according-to-the-reviewer’s-comments`. The filenames within the repository are descriptive and indicate the specific example they correspond to.

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


