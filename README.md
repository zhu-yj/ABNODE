# ABNODE

## Introduction
**ABNODE** (Auto-tuning Blimp-oriented Neural Ordinary Differential Equation) is a hybrid modeling framework that integrates first-principle physical modeling with neural networks for accurately capturing the dynamic behavior of robotic blimps. This repository provides all the necessary data, code, and scripts to train and test ABNODE as well as comparative models like BNODE, KNODE, and SINDYc.

The repository is modularly structured, allowing users to easily navigate the components, run experiments, and extend the models.

## Folder Structure

```
├── data                # Neural ODE and dynamics model implementations
│   ├── data_1.csv      # data_1 file
│   ├── ...             # data files
│   ├── data_info       # Information about the dataset
│   └── readme.txt      # Description of the data files
├── logs                # Logs generated during model training/testing
├── methods             # Training algorithms for different models
│   ├── ABNODE_phase1.py # ABNODE training script (Phase 1)
│   ├── ABNODE_phase2.py # ABNODE training script (Phase 2)
│   ├── comp_BNODE.py   # Comparative BNODE training script
│   └── ...             # Other training scripts
├── models              # Neural ODE and dynamics model implementations
│   ├── NODE_MLP.py     # NODE model using MLP architecture
│   ├── RGBlimp_dynamics.py  # Blimp dynamics model
│   └── ...             # Other model files
├── record              # Generated data during experiments
├── sh                  # Shell scripts for automation
│   ├── abnode_0.sh     # Example training script
│   └── ...             # Additional shell scripts
├── utils               # Helper functions and utilities
│   ├── parameters.py   # Physical parameters configuration
│   ├── solvers.py      # ODE solvers and numerical methods
│   └── ...
└── requirements.txt    # List of dependencies required to run the code
```

### Folder Descriptions

#### `data/`
This folder contains trajectory data for the RGBlimp, which includes various measurements such as position, Euler angles, velocity, and angular velocity.

#### `methods/`
This folder contains the Python scripts used to implement and train various models:
- **ABNODE_phase1.py**: Implements Phase 1 of the ABNODE model training.
- **ABNODE_phase2.py**: Implements Phase 2 of the ABNODE model training.
- **comp_BNODE.py**: Script for training and comparing the BNODE model.
- **comp_KNODE.py**: Script for training the KNODE model.
- **comp_NODE.py**: Script for training the NODE model.
- **comp_SINDYc.py**: Script for training the SINDYc model, which is used for sparse identification of dynamics.

#### `models/`
This folder contains the models for different methods.
- **NODE_MLP.py**: Neural ODE with a Multilayer Perceptron (MLP) architecture.
- **RGBlimp_dynamics_ABNODE_p1.py**: The dynamic model for RGBlimp used in Phase 1 of ABNODE.
- **RGBlimp_dynamics_ABNODE_p2.py**: The dynamic model used in Phase 2 of ABNODE.
- **RGBlimp_dynamics_KNODE.py**: The dynamics model used in the KNODE approach.
- **RGBlimp_dynamics.py**: The core dynamics model for the RGBlimp, common across several methods.

#### `utils/`
Utility functions and helper scripts to assist with model development:
- **NODE.py**: Contains functions related to NODE.
- **parameters.py**: Manages physical parameters for the blimp models.
- **skew.py**: Implements the skew matrix calculation used in different models.
- **solvers.py**: Implements numerical solvers for integrating ODEs.

#### `requirements.txt`
Contains the list of dependencies and their versions. To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Usage
To start training the ABNODE model, simply run the corresponding shell script:
```bash
sh ./sh/abnode_0.sh
```
Logs and results will be saved in the `logs/` and `record/` folders, respectively.


