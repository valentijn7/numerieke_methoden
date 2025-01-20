# numerieke_methoden/III/algorithms.py

from typing import List, Dict, Tuple, Any, Callable
import numpy as np


def forward_euler(
        current_state: np.array,
        current_time: float,
        step_size: float,
        derivative: callable,
        C: Any
    ) -> np.array:
    """ Performs one step of the Forward Euler method
    
    :param current_state: current state of the system (dependent variable)
    :param current_time: current time of the system (independent variable)
    :param step_size: step size of the integration
    :param derivative: function that returns derivatives for current state and time
    :param C: class (struct) with constants
    :return: updated values of dependent variable at next time step
    """
    return current_state + step_size * derivative(current_state, current_time, C)


def leap_frog(
        prev_state: np.array,
        current_state: np.array,
        current_time: float,
        step_size: float,
        derivative: callable,
        C: Any
    ) -> np.array:
    """ Performs one step of the Leap Frog method

    :param prev_state: previous state of the system (dependent variable)
    :param current_state: current state of the system (dependent variable)
    :param current_time: current time of the system (independent variable)
    :param step_size: step size of the integration
    :param derivative: function that returns derivatives for current state and time
    :param C: class (struct) with constants
    :return: updated values of dependent variable at next time step
    """
    return prev_state + 2 * step_size * derivative(current_state, current_time, C)


def adam_bashforth(
        prev_state: np.array,
        current_state: np.array,
        current_time: float,
        step_size: float,
        derivative: callable,
        C: Any
    ) -> np.array:
    """
    Performs one step of the Adams-Bashforth method

    :param prev_state: previous state of the system (dependent variable)
    :param current_state: current state of the system (dependent variable)
    :param current_time: current time of the system (independent variable)
    :param step_size: step size of the integration
    :param derivative: function that returns derivatives for current state and time
    :param C: class (struct) with constants
    :return: updated values of dependent variable at next time step
    """
    return current_state + step_size * \
        (1.5 * derivative(current_state, current_time, C) \
         - 0.5 * derivative(prev_state, current_time, C))


def init_crank_nicolson(
        C: Any
) -> Tuple[np.array, np.array]:
    """
    Init, precompute, and return the matrices for the Crank-nicolson 

    Assumption: Dirichlet boundary conditions

    :param C: class (struct) with constants
    :return: matrices A and B
    """
    n = C.n_x + 1
    A = np.zeros((n, n), dtype = float)
    B = np.zeros((n, n), dtype = float)

    sigma = C.kappa * C.dt / (C.dx ** 2)
                            # init main- and off-diagonal elements, rest is zero
    for idx in range(1, n - 1):
        A[idx, idx] = 1 + sigma
        A[idx, idx - 1] = -sigma / 2
        A[idx, idx + 1] = -sigma / 2

        B[idx, idx] = 1 - sigma
        B[idx, idx - 1] = sigma / 2
        B[idx, idx + 1] = sigma / 2

                            # Dirichlet boundary conditions:
    A[0, :] = 0             # zero out first row;
    A[0, 0] = 1             # add 1 on diagonal
    B[0, :] = 0
    B[0, 0] = 1

    A[-1, :] = 0            # same for final row
    A[-1, -1] = 1
    B[-1, :] = 0
    B[-1, -1] = 1

    return A, B


def crank_nicolson(
        current_state: np.array,
        A: np.array,
        B: np.array,
        constants: Any
) -> np.array:
    """
    Performs one step of the Crank-Nicolson method (with Dirichlet)

    :param current_state: current state of the system (dependent variable)
    :param A: matrix A
    :param B: matrix B
    :param constants: class (struct) with constants
    :return: update to next time step
    """
    intermediate = np.dot(B, current_state)
    intermediate[0], intermediate[-1] = constants.T1, constants.T0
    return np.linalg.solve(A, intermediate)

    
def runge_kutta_4(
        current_state: np.array,
        current_time: float,
        step_size: float,
        derivative: callable,
        constants: Any
    ) -> np.array:
    """ Performs one step of the 4th order Runge-Kutta
    
    :param current_state: current state of the system (dependent variable)
    :param current_time: current time of the system (independent variable)
    :param step_size: step size of the integration
    :param derivative: function that returns derivatives for current state and time
    :param constants: class (struct) with constants
    :return: updated values of dependent variable at next time step
    """
                            # calculate the 4 k's
    k1 = step_size * derivative(current_state, current_time, constants)
    k2 = step_size * derivative(current_state + 0.5 * k1, current_time + 0.5 * step_size, constants)
    k3 = step_size * derivative(current_state + 0.5 * k2, current_time + 0.5 * step_size, constants)
    k4 = step_size * derivative(current_state + k3, current_time + step_size, constants)
                            # calculate and return new state
    return current_state + (k1 + 2 * k2 + 2 * k3 + k4) / 6