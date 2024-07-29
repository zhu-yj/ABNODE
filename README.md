# RGBlimp Trajectory Dataset

## Introduction
This is a comprehensive dataset containing RGBlimp trajectory data, which includes position, Euler angles, velocity, angular velocity, and more. This dataset is ideal for studying and analyzing the dynamic behavior of miniature robotic blimps.

## File Description
The dataset includes 140 trajectory data points.
- The index ranges from 0 to 34, and each index contains four trajectory data files.
- The mapping from index to data names is `index * 4 + 1` to `index * 4 + 4`.

### Input
The relationship between index and input is illustrated in the table below.

## Data Fields
### position_data
- `p_1`: X-axis position coordinate (unit: meters)
- `p_2`: Y-axis position coordinate (unit: meters)
- `p_3`: Z-axis position coordinate (unit: meters)

### euler_angle
- `e_1`: Roll angle (unit: degrees)
- `e_2`: Pitch angle (unit: degrees)
- `e_3`: Yaw angle (unit: degrees)

### velocity_data
- `vb_1`: Velocity in the X direction in body frame (unit: meters/second)
- `vb_2`: Velocity in the Y direction in body frame (unit: meters/second)
- `vb_3`: Velocity in the Z direction in body frame (unit: meters/second)

### angular_velocity
- `wb_1`: Angular velocity in the X direction in body frame (unit: rad/s)
- `wb_2`: Angular velocity in the Y direction in body frame (unit: rad/s)
- `wb_3`: Angular velocity in the Z direction in body frame (unit: rad/s)

## Usage Example
Below is a simple Python example demonstrating how to load and use the data:

```python
import pandas as pd

# Load trajectory data
trajectory_data = pd.read_csv('data_1.csv')
print("Trajectory Data Preview:")
print(trajectory_data.head())
