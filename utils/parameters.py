import numpy as np


'''
RGBlimp model parameters
'''
RGBlimp_params = {
    'Rho': 1.2187,  # Air density (kg/m³)
    'rho': 0.1690,  

    'g': 9.8000,  # Acceleration due to gravity (m/s²)
    'B': 0.15204,  # Buoyant force
    'm': 0.10482, 
    'mb': 0.05407,  
    'G': 0.06713,  # Gravitational force

    'r': [[-0.0432], [-0.0003], [0.0079]],  # Vector representing the center of gravity (m)
    'rb': [[0.0747], [0.0006], [0.2380]],  # Vector representing the position of the gondola (m)
    'd': 0.150,  
    'A': 0.250,  # Reference area for aerodynamic forces (m²)

    'Ix': 0.0300,  # Moment of inertia about the x-axis (kg·m²)
    'Iy': 0.0150,  # Moment of inertia about the y-axis (kg·m²)
    'Iz': 0.0100,  # Moment of inertia about the z-axis (kg·m²)
    'I': np.diag([0.0300, 0.0150, 0.0100]),  # Inertia matrix (diagonal for simplicity)

    'K1': -0.0503, 
    'K2': -0.0264, 
    'K3': -0.0137,  

    'Cd0': 0.2425,  # Base drag coefficient
    'Cda': 4.4195,  # Drag coefficient derivative with respect to angle of attack
    'Cdb': 7.5080,  # Drag coefficient derivative with respect to side-slip angle
    'Cs0': 0.0083,  # Base side force coefficient
    'Csa': -0.0744, # Side force coefficient derivative with respect to angle of attack
    'Csb': -2.1140, # Side force coefficient derivative with respect to side-slip angle
    'Cl0': 0.1594,  # Base lift coefficient
    'Cla': 2.9375,  # Lift coefficient derivative with respect to angle of attack
    'Clb': 4.5537,  # Lift coefficient derivative with respect to side-slip angle

    'Cmx0': 0.0131, # Base rolling moment coefficient
    'Cmxa': -0.0301,# Rolling moment coefficient derivative with respect to angle of attack
    'Cmxb': -0.5256,# Rolling moment coefficient derivative with respect to side-slip angle
    'Cmy0': 0.0568, # Base pitching moment coefficient
    'Cmya': 0.0933, # Pitching moment coefficient derivative with respect to angle of attack
    'Cmyb': 5.2357, # Pitching moment coefficient derivative with respect to side-slip angle
    'Cmz0': 0.0006, # Base yawing moment coefficient
    'Cmza': -0.0012,# Yawing moment coefficient derivative with respect to angle of attack
    'Cmzb': -0.0936,# Yawing moment coefficient derivative with respect to side-slip angle
}