# ABNODE

## Introduction
**ABNODE** (Auto-tuning Blimp-oriented Neural Ordinary Differential Equation) is a hybrid modeling framework that combines first-principle physics and neural networks to create accurate models of blimp dynamics. This method is designed to enhance the prediction capabilities for trajectory tracking, control, and other dynamic behaviors of miniature robotic blimps.

This repository contains the datasets, source code, and scripts required to reproduce the results presented in our research. The implementation is structured in a modular way to facilitate easy use, modification, and extension.

## Folder Structure

```
├── data                # Experimental data and descriptions
│   ├── data_1.csv      # Example data file
│   ├── ...             # Additional data files
│   ├── data_info       # Information about the dataset
│   └── readme.txt      # Description of data usage and structure
├── logs                # Logs generated during training/testing
├── methods             # Training algorithms for various models
│   ├── ABNODE_phase1.py
│   ├── ABNODE_phase2.py
│   ├── comp_BNODE.py   # Comparative BNODE training script
│   └── ...             # Other scripts
├── models              # Predefined models and dynamics
│   ├── NODE_MLP.py     # NODE model using MLP architecture
│   ├── RGBlimp_dynamics.py  # Blimp dynamics model
│   └── ...             # Other model files
├── record              # Generated data during experiments
├── sh                  # Shell scripts for automation
│   ├── abnode_0.sh     # Example training script
│   └── ...             # Additional shell scripts
├── utils               # Helper functions and utilities
│   ├── parameters.py   # Parameter handling
│   ├── solvers.py      # ODE solvers and numerical methods
│   └── ...
└── requirements.txt    # Python package dependencies
```

### Folder Description

#### `data/`
Contains the RGBlimp trajectory dataset used for training and testing. The data include measurements of position, Euler angles, velocity, angular velocity, and other relevant parameters for modeling the blimp's dynamic behavior.

- **File Description**: 140 trajectory data points, organized in sets of four per index.
  - Example mapping: for `index = 0`, data files are from `1` to `4`, and for `index = 1`, files are from `5` to `8`.

#### `methods/`
This folder contains the implementation of training algorithms for several modeling approaches:
- **ABNODE**: Auto-tuning Neural ODE for blimps
- **BNODE**: Baseline Neural ODE
- **KNODE**: Physics-informed NODE with known dynamics
- **SINDYc**: Sparse Identification of Nonlinear Dynamics (control version)

#### `models/`
Predefined models for simulating and training various ODE-based methods:
- **ABNODE model**: Core model that combines physics and neural networks.
- **RGBlimp dynamics**: Blimp-specific dynamic model to simulate real-world behaviors.

#### `record/`
Contains the results and metrics generated during the training and testing processes.

#### `sh/`
Shell scripts for automated training, testing, and evaluation of the models. These scripts streamline the process of running experiments with predefined parameters.

#### `utils/`
Utility scripts that contain helper functions, such as ODE solvers, parameter configurations, and other shared functions across models.

#### `requirements.txt`
Lists the Python dependencies required to run the code. To install, simply run:
```bash
pip install -r requirements.txt
```

## Usage

To begin training the ABNODE model, execute the following shell script:
```bash
sh ./sh/abnode_0.sh
```

Make sure to customize the script parameters as needed for your experiment. All training logs and model checkpoints will be saved in the `logs/` and `record/` directories.

