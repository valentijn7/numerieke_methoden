# numerieke_methoden/III/diffusion/algorithms.py

from typing import List, Dict, Tuple, Any, Callable
import numpy as np


# For the SWE, we only use Runge-Kutta 4th order.

    
def RK4(
        state: np.ndarray,
        h: np.ndarray,
        v: np.ndarray,
        b: np.ndarray,
        dt: float,
        dx: float,
        derivative: callable,
        C: Any
    ) -> np.array:
    """ Performs one step of the 4th order Runge-Kutta
    
    :param state: current state of the system (either height or velocity)
    :param h: current height of the system
    :param v: current velocity of the system 
    :param b : current rest height of water
    :param dt: step size of time iteration
    :param dx: step size of space iteration
    :param derivative: function that returns derivatives for current state and time
    :param C: class (struct) with constants
    :return: updated values of dependent variable at next time step
    """
                            # calculate the 4 k's
    k1 = dt * derivative(h, v, b, dt, dx, C)
    k2 = dt * derivative(h + 0.5 * k1, v + 0.5 * dt * k1, b, dt, dx, C)
    k3 = dt * derivative(h + 0.5 * k2, v + 0.5 * dt * k2, b, dt, dx, C)
    k4 = dt * derivative(h + k3, v + k3, b, dt, dx, C)
                            # calculate and return new state
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6