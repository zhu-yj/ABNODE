# -*- coding: UTF-8 -*-

import numpy as np

'''
ODE Solvers
'''

def RK_RGB(z0, t_total, f, step_size):
    '''
    Runge-Kutta ODE solver
    Input:
      z0: Initial condition (state)
      t_total: Total time for integration
      f: ODE function that computes the derivative of the state
      step_size: Time step for integration
    Return:
      z: The state after one integration step
    '''
    z = z0

    # Compute Runge-Kutta intermediate steps
    k1 = f(z0)
    k2 = f(z0 + 0.5 * step_size * k1)
    k3 = f(z0 + 0.5 * step_size * k2)
    k4 = f(z0 + step_size * k3)

    # Update the state using the weighted sum of the slopes
    z = z0 + (1 / 6.0) * step_size * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Handle Euler angles, specifically adjusting psi to be within [-pi, pi]
    if z[0, 5] > np.pi:
        z[0, 5] = z[0, 5] - 2 * np.pi
    elif z[0, 5] < -np.pi:
        z[0, 5] = z[0, 5] + 2 * np.pi

    return z


def Euler(z0, n_steps, f, step_size):
    '''
    Simplest Euler ODE initial value solver
    Input:
      z0: Initial condition (state)
      n_steps: The number of steps for integration
      f: ODE function that computes the derivative of the state
      step_size: Time step for integration
    Return:
      z: The state after n_steps of integration
    '''
    z = z0  # Initialize state with the initial condition

    # Iterate over the number of integration steps
    for i_step in range(int(n_steps)):
        # Update the state using Euler's method
        z = z + step_size * f(z)
    
    return z