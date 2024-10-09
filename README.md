# ABNODE

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

### data
This is a comprehensive dataset containing RGBlimp trajectory data, which includes position, Euler angles, velocity, angular velocity, and more. This dataset is ideal for studying and analyzing the dynamic behavior of miniature robotic blimps.
#### File Description
The dataset includes 140 trajectory data points.
- The index ranges from 0 to 34, and each index contains four trajectory data files.
- The mapping from index to data names is `index * 4 + 1` to `index * 4 + 4`.
#### Input
The relationship between index and input is illustrated in the table below.
| rdx \[cm\]  | (1.4,5.8) \[gf\] | (1.6,5.5) \[gf\] |**&#9474;**| (1.2,6.1) \[gf\] | (1.7,6.1) \[gf\] | (1.2,5.4) \[gf\] | (1.7,5.4) \[gf\] | (2.05,2.05) \[gf\] | rdx \[cm\]  |
|------|-----------|-----------|---|-----------|-----------|-----------|-----------|-------------|------|
| 0    | 25        | 20        | **&#9474;** |15        | 10        | 5         | 0         | 30          | 0    |
| 1 | 26        | 21        | **&#9474;** |16        | 11        | 6         | 1         | 31          | 1 |
| 2 | 27        | 22        | **&#9474;** |17        | 12        | 7         | 2         | 32          | 2 |
| 3 | 28        | 23        | **&#9474;** |18        | 13        | 8         | 3         | 33          | 3 |
| 4 | 29        | 24        | **&#9474;** |19        | 14        | 9         | 4         | 34          | -1 |

### methods
contains the ABNODE, BNODE, KNODE, NODE, SINDYc training algorithm implementation

### models
contains the ABNODE model, the RGBlimp dynamics model, and so on.  

### record
record the data generated during training and testing

### sh
.sh scripts used for training

### utils
Ode solvers and physical parameters

### requirements.txt

## Usage Example
``` shell
sh ./sh/abnode_0.sh
```


