# RGBlimp Trajectory Dataset

## Introduction
This is a comprehensive dataset containing RGBlimp trajectory data, which includes position, Euler angles, velocity, angular velocity, and more. This dataset is ideal for studying and analyzing the dynamic behavior of miniature robotic blimps.

## File Description
The dataset includes 140 trajectory data points.
- The index ranges from 0 to 34, and each index contains four trajectory data files.
- The mapping from index to data names is `index * 4 + 1` to `index * 4 + 4`.

### Input
The relationship between index and input is illustrated in the table below.
<table>
  <tr>
    <th>rdx</th>
    <th>(1.4,5.8)</th>
    <th>(1.6,5.5)</th>
    <th>(1.2,6.1)</th>
    <th>(1.7,6.1)</th>
    <th>(1.2,5.4)</th>
    <th>(1.7,5.4)</th>
    <th>(2.05,2.05)</th>
    <th>rdx</th>
  </tr>
  <tr>
    <td>0</td>
    <td style="background-color: lightblue;">25</td>
    <td style="background-color: lightblue;">20</td>
    <td style="background-color: lightblue;">15</td>
    <td style="background-color: lightgreen;">10</td>
    <td style="background-color: lightgreen;">5</td>
    <td style="background-color: lightgreen;">0</td>
    <td>30</td>
    <td>0</td>
  </tr>
  <tr>
    <td>0.01</td>
    <td style="background-color: lightblue;">26</td>
    <td style="background-color: lightblue;">21</td>
    <td style="background-color: lightblue;">16</td>
    <td style="background-color: lightgreen;">11</td>
    <td style="background-color: lightgreen;">6</td>
    <td style="background-color: lightgreen;">1</td>
    <td>31</td>
    <td>0.01</td>
  </tr>
  <tr>
    <td>0.02</td>
    <td style="background-color: lightblue;">27</td>
    <td style="background-color: lightblue;">22</td>
    <td style="background-color: lightblue;">17</td>
    <td style="background-color: lightgreen;">12</td>
    <td style="background-color: lightgreen;">7</td>
    <td style="background-color: lightgreen;">2</td>
    <td>32</td>
    <td>0.02</td>
  </tr>
  <tr>
    <td>0.03</td>
    <td style="background-color: lightblue;">28</td>
    <td style="background-color: lightblue;">23</td>
    <td style="background-color: lightblue;">18</td>
    <td style="background-color: lightgreen;">13</td>
    <td style="background-color: lightgreen;">8</td>
    <td style="background-color: lightgreen;">3</td>
    <td>33</td>
    <td>0.03</td>
  </tr>
  <tr>
    <td>0.04</td>
    <td style="background-color: lightblue;">29</td>
    <td style="background-color: lightblue;">24</td>
    <td style="background-color: lightblue;">19</td>
    <td style="background-color: lightgreen;">14</td>
    <td style="background-color: lightgreen;">9</td>
    <td style="background-color: lightgreen;">4</td>
    <td>34</td>
    <td>-0.01</td>
  </tr>
</table>
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
