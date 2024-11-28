
### Data and Code for Images and Algorithms in the Paper "Online Physics-Informed Dynamic Mode Decomposition: Theory and Applications"

#### Introduction
This document provides a detailed description of the data, images, and algorithms used in the paper "Online Physics-Informed Dynamic Mode Decomposition: Theory and Applications". The purpose is to ensure reproducibility and facilitate a deeper understanding of the methodologies employed.

### Updates Based on Reviewer Comments

In response to the first round of review comments, we have added several new examples. The corresponding code for these examples has been placed in a new repository titled `banach-Here’s-the-code-modified-according-to-the-reviewer’s-comments`. The filenames within the repository are descriptive and indicate the specific example they correspond to.

With the exception of the ‘Cylinder Flow’ example, the data for all other examples can be directly generated using the provided code, so there is no need to upload additional data. For the ‘Cylinder Flow’ example, the data can be downloaded independently from [https://www.databookuw.com/](https://www.databookuw.com/).


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


