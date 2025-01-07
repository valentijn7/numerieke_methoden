#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:16:29 2024

@author: Valentijn Oldenburg
"""
from typing import List, Dict, Tuple, Any
import numpy as np
# 1) add here the setting(s) you run the model

# but define the physical constants in this constants_class:
# Again, this is required to unify scripts to ease correcting it
class PhysConstants:
    def __init__(self):
        self.Kappa  = 1       # Thermal diffusion coefficient (m2/s)
        self.T0     = 273      # Initial temperature rod (K)
        self.T1     = 373      # Temperature of rod at x=0 for t>0
        self.L      = 10        # Length of the rod (m)
        self.n_x     = 100       # Number of gridpoints in the rod
        self.t_total = 1000   # Total time of the simulation (s)
        self.dt     = 1         # Time step of the simulation (s)
        self.n_t   = int(self.t_total / self.dt)  # Number of time steps


# 2) add here supporting code 
def RK4(
        current_state: np.array,
        current_time: float,
        step_size: float,
        derivative: callable,
        d_c: Dict[str, float]
    ) -> np.array:
    """ Performs one step of the 4th order Runge-Kutta
    
    :param current_state: current state of the system (dependent variable)
    :param current_time: current time of the system (independent variable)
    :param step_size: step size of the integration
    :param derivative: function that returns derivatives for current state and time
    :param d_c: dictionary with constants
    :return: updated values of dependent variable at next time step
    """
                            # calculate the 4 k's
    k1 = step_size * derivative(current_state, current_time, d_c)
    k2 = step_size * derivative(current_state + 0.5 * k1, current_time + 0.5 * step_size, d_c)
    k3 = step_size * derivative(current_state + 0.5 * k2, current_time + 0.5 * step_size, d_c)
    k4 = step_size * derivative(current_state + k3, current_time + step_size, d_c)

                            # calculate and return new state
    return current_state + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def forward_euler(
        current_state: np.array,
        current_time: float,
        step_size: float,
        derivative: callable,
        d_c: Dict[str, float]
    ) -> np.array:
    """ Performs one step of the Forward Euler method
    
    :param current_state: current state of the system (dependent variable)
    :param current_time: current time of the system (independent variable)
    :param step_size: step size of the integration
    :param derivative: function that returns derivatives for current state and time
    :param d_c: dictionary with constants
    :return: updated values of dependent variable at next time step
    """
    slope = derivative(current_state, current_time, d_c)
    return current_state + step_size * slope


def central_derivative(
        array: np.array,
        idx: int) -> float:
    """ Calculate the central derivative of an array at a given index
    
    :param array: array to calculate the derivative of
    :param idx: index to calculate the derivative at
    :return: central derivative of the array at the given index
    """
    if idx == 0 or idx == len(array) - 1:
        raise ValueError("Index cannot be at the edge of the array")

    # Klopt nog niet

    return (array[idx + 1] - array[idx - 1]) / 2

def derivative(

)
    """
    Calculate the heat equation




    """




# 3) Use the following routine to start simulations
def Task1_caller(L, nx, TotalTime, dt, TimeSteppingMethod, 
                 DiffMethod="CD"):
    # The mandatory input is:
    # L                   Length of domain to be modelled (m)
    # nx                  Number of gridpoint in the model domain
    # TotalTime           Total length of the simulation (s)
    # dt                  Length of each time step (s)
    # TimeSteppingMethod  Could be:
    #  "Theory"             Theoretical solution
    #  "AB"                 Adams-Bashforth
    #  "CN"                 Crank-Nicholson
    #  "EF"                 Euler Forward
    #  "LF"                 Leaf Frog
    #  "RK4"                Runge-Kutta 4
    #
    # The optional input is:
    # DiffMethod  Method to determine the 2nd order spatial derivative
    #   Default = "CD"    Central differences
    #    Option = "PS"    Pseudo spectral
    # 
    # The output is:
    # Time        a 1-D array (length nt) with time values considered
    # Xaxis       a 1-D array (length nx) with x-values used
    # Result      a 2-D array (size [nx, nt]), with the results of the routine    
    # You may add extra output after these three
    
    C = PhysConstants()       # load physical constants in `self`-defined variable PhysC
    # First, we define a grid, or matrix, for space (horizontal) and time (vertical);
    # second, fill the first row with the initial temperature of the rod;
    # third, add the boundary conditions for the rod at x = 0 for t > 0
    grid = np.zeros(shape = (int(C.n_x, C.n_t)), dtype = float)
    grid[:, 0] = C.T0
    grid[0, :] = C.T1


    
    return Time, Xaxis, Result    


# 4) Code here (or in another file) the commands that use Task1_caller

# 5) Code here (or in another file) the analysis of the results and the plotting

def main():
    pass


if __name__ == "__main__":