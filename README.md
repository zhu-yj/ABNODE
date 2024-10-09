在基于你提供的文件结构基础上，进一步扩展和完善 `README.md` 文件，保持简洁但增强对各个文件和目录的解释。以下是更新后的版本：

---

# ABNODE

## Introduction
**ABNODE** (Auto-tuning Blimp-oriented Neural Ordinary Differential Equation) is a hybrid modeling framework that integrates first-principle physical modeling with neural networks for accurately capturing the dynamic behavior of robotic blimps. This repository provides all the necessary data, code, and scripts to train and test ABNODE as well as comparative models like BNODE, KNODE, and SINDYc.

The repository is modularly structured, allowing users to easily navigate the components, run experiments, and extend the models.

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

### Folder Descriptions

#### `data/`
This folder contains trajectory data for the RGBlimp, which includes various measurements such as position, Euler angles, velocity, and angular velocity. It is divided into multiple CSV files, with each file representing different trajectory points.

- **data_info_spiral.csv**: Metadata for spiral trajectories.
- **data_info_straight.csv**: Metadata for straight-line trajectories.
- **readme.txt**: Additional documentation explaining the structure and content of the dataset.

#### `logs/`
This folder stores the logs generated during model training and testing. Each log captures important information about the training process, such as loss curves, model checkpoints, and hyperparameters used.

#### `methods/`
This folder contains the Python scripts used to implement and train various models:
- **ABNODE_phase1.py**: Implements Phase 1 of the ABNODE model training.
- **ABNODE_phase2.py**: Implements Phase 2 of ABNODE, focusing on fine-tuning and further optimizations.
- **comp_BNODE.py**: Script for training and comparing the BNODE model.
- **comp_KNODE.py**: Script for training the physics-informed KNODE.
- **comp_NODE.py**: Vanilla NODE model training script.
- **comp_SINDYc.py**: Script for training the SINDYc model, which is used for sparse identification of dynamics.

#### `models/`
This folder contains the actual model implementations:
- **NODE_MLP.py**: Neural ODE with a Multilayer Perceptron (MLP) architecture.
- **RGBlimp_dynamics_ABNODE_p1.py**: The dynamic model for RGBlimp used in Phase 1 of ABNODE.
- **RGBlimp_dynamics_ABNODE_p2.py**: The dynamic model used in Phase 2 of ABNODE.
- **RGBlimp_dynamics_KNODE.py**: The dynamics model used in the KNODE approach.
- **RGBlimp_dynamics.py**: The core dynamics model for the RGBlimp, common across several methods.

#### `record/`
Stores the results generated during training and testing:
- Each subdirectory (e.g., `abnode`, `bnode`, `knode`, etc.) corresponds to records from specific model experiments.

#### `sh/`
Shell scripts to automate the training and evaluation of the models. Each script is tailored to a specific model:
- **abnode_0.sh**: Runs the ABNODE training pipeline.
- **bnode_0.sh**: Runs the BNODE comparison model.
- **knode_0.sh**: Runs the KNODE model.
- **node_0.sh**: Runs the vanilla NODE model.
- **sindy_0.sh**: Runs the SINDYc model.

#### `utils/`
Utility functions and helper scripts to assist with model development:
- **NODE.py**: Contains helper functions related to NODE.
- **parameters.py**: Manages physical parameters for the blimp models.
- **print_package_version.py**: Prints the versions of the installed packages.
- **skew.py**: Implements the skew matrix calculation used in certain ODE solvers.
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

## Contributing

We welcome contributions! If you find any issues or have ideas for improvements, feel free to submit a pull request or open an issue.

---

这样调整后，`README.md` 更详细地解释了项目结构中每个文件夹和文件的功能，使用户可以更容易理解代码库内容，配置环境并运行实验。
