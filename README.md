# RGBlimp Trajectory Dataset

## Introduction
The Auto-tuning Blimp-oriented Neural Ordinary Differential Equation method (ABNODE), a data-driven approach that integrates first-principle and neural network modeling.
## Folder Description
```
├── data
│   ├── data_1.csv
│   ├── ...
│   ├── data_info
│   └── readme.txt
├── logs
├── methods
│   ├── ABNODE_phase1.py
│   ├── ABNODE_phase2.py
│   ├── comp_BNODE.py
│   └── ...
├── models
│   ├── NODE_MLP.py
│   ├── RGBlimp_dynamics.py
│   └── ...
├── record
├── sh
│   ├── abnode_0.sh
│   └── ...
├── utils
│   ├── parameters.py
│   ├── solvers.py
│   └── ...
└── requirements.txt
```

### ./data/
This is a comprehensive dataset containing RGBlimp trajectory data, which includes position, Euler angles, velocity, angular velocity, and more. This dataset is ideal for studying and analyzing the dynamic behavior of miniature robotic blimps.
#### File Description
The dataset includes 140 trajectory data points.
- The index ranges from 0 to 34, and each index contains four trajectory data files.
- The mapping from index to data names is `index * 4 + 1` to `index * 4 + 4`.
#### Data Files
All dataset files are stored in the `data` directory. This includes:
- `data/data_1.csv`: (rdx, Fl, Fr)=(0, 1.7, 5.4)
- `data/data_2.csv`: (rdx, Fl, Fr)=(0, 1.7, 5.4)
- ...
- `data/data_4.csv`: (rdx, Fl, Fr)=(-1, 2.05, 2.05)

#### Input
The relationship between index and input is illustrated in the table below.
| rdx \[cm\]  | (1.4,5.8) \[gf\] | (1.6,5.5) \[gf\] |**&#9474;**| (1.2,6.1) \[gf\] | (1.7,6.1) \[gf\] | (1.2,5.4) \[gf\] | (1.7,5.4) \[gf\] | (2.05,2.05) \[gf\] | rdx \[cm\]  |
|------|-----------|-----------|---|-----------|-----------|-----------|-----------|-------------|------|
| 0    | 25        | 20        | **&#9474;** |15        | 10        | 5         | 0         | 30          | 0    |
| 1 | 26        | 21        | **&#9474;** |16        | 11        | 6         | 1         | 31          | 1 |
| 2 | 27        | 22        | **&#9474;** |17        | 12        | 7         | 2         | 32          | 2 |
| 3 | 28        | 23        | **&#9474;** |18        | 13        | 8         | 3         | 33          | 3 |
| 4 | 29        | 24        | **&#9474;** |19        | 14        | 9         | 4         | 34          | -1 |

#### Data Fields
##### position_data
- `p_1`: X-axis position coordinate (unit: meters)
- `p_2`: Y-axis position coordinate (unit: meters)
- `p_3`: Z-axis position coordinate (unit: meters)

##### euler_angle
- `e_1`: Roll angle (unit: degrees)
- `e_2`: Pitch angle (unit: degrees)
- `e_3`: Yaw angle (unit: degrees)

##### velocity_data
- `vb_1`: Velocity in the X direction in body frame (unit: meters/second)
- `vb_2`: Velocity in the Y direction in body frame (unit: meters/second)
- `vb_3`: Velocity in the Z direction in body frame (unit: meters/second)

##### angular_velocity
- `wb_1`: Angular velocity in the X direction in body frame (unit: rad/s)
- `wb_2`: Angular velocity in the Y direction in body frame (unit: rad/s)
- `wb_3`: Angular velocity in the Z direction in body frame (unit: rad/s)
other data field, see the files for details

#### Usage Example
Below is a simple Python example demonstrating how to load and use the data:

```python
import pandas as pd

# Load trajectory data
trajectory_data = pd.read_csv('data_1.csv')
print("Trajectory Data Preview:")
print(trajectory_data.head())
```

### ./methods/
contains the ABNODE, BNODE, KNODE, NODE, SINDYc training algorithm implementation
### ./models/
contains the ABNODE model, the RGBlimp dynamics model, and so on.  
### ./record/
record the data generated during training and testing
### ./sh/
.sh scripts used for training
### ./utils/
Ode solvers and physical parameters
### requirements.txt
## Usage Example
``` shell
sh ./sh/abnode_0.sh
```


