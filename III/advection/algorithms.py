# numerieke_methoden/III/advection/algorithms.py

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
    ) -> Tuple[np.ndarray, np.ndarray]:
    """ Creates and returns the matrices A and B for the Crank-Nicolson.
    This time, with periodic boundary conditions for 1D advection.
    
    :param C: class (struct) with constants
    :return: matrices A and B
    """
    n = C.n_x
                            # we init A and B 
    A = np.zeros((n, n), dtype = float)
    B = np.zeros((n, n), dtype = float)
                            # we calculate sigma (for advection)
    sigma = C.c * C.dt / C.dx  
    
    for idx in range(n):    # we fill the matrices
        A[idx, idx] = 1     # diagonal elements are 1
        B[idx, idx] = 1
                            # off-diagonal elements are
                            # sigma / 4 and -sigma / 4
        idx_minus = (idx - 1) % n
        idx_plus  = (idx + 1) % n

        A[idx, idx_minus] -= sigma / 4
        A[idx, idx_plus]  += sigma / 4

        B[idx, idx_minus] += sigma / 4
        B[idx, idx_plus]  -= sigma / 4

    return A, B


def crank_nicolson(
        current_state: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        C: Any
    ) -> np.ndarray:
    """ One Crank-Nicolson step with periodic boundary conditions

    :param current_state: current state of system
    :param A: matrix A
    :param B: matrix B
    :param C: class (struct) with constants
    :return: updated state (next time step)
    """
    # https://stackoverflow.com/questions/34142485/difference-between-numpy-dot-and-python-3-5-matrix-multiplication
    return np.linalg.solve(A, np.matmul(B ,current_state))

    
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