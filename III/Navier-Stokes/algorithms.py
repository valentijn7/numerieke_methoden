# numerieke_methoden/III/diffusion/algorithms.py

from typing import List, Dict, Tuple, Any, Callable
import numpy as np

from typing import List, Dict, Tuple, Any, Callable
import numpy as np


'''    
def forward_euler(
        current_height: np.ndarray,
        current_velocity: np.ndarray,
        current_b: np.ndarray,
        dt: float,
        derivative: callable,
        C: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
    """ Performs one step of the Forward Euler method
    # TODO! misschien toch nog terugveranderen !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    :param current_state: current state of the system 
    :param current_velocity: current velocity of the system 
    :param current_b : current rest height of water
    :param dt: step size of time iteration
    :param derivative: function that returns derivatives for current state and time
    :param C: class (struct) with constants
    :return: updated values of dependent variable at next time step
    """
    dH_dt = derivative(current_height, current_velocity,current_b,C.dt,C.dx,C)[0]
    du_dt = derivative(current_height, current_velocity,current_b,C.dt,C.dx,C)[1]
    
    H_new = current_height + dt * dH_dt
    u_new = current_velocity + dt * du_dt
    
    return (H_new, u_new)
'''

def forward_euler(# 8 arguments
        current_state: np.ndarray,
        current_height: np.ndarray,
        current_velocity: np.ndarray,
        current_b: np.ndarray,
        dt: float,
        dx: float,
        derivative: callable,
        C: Any
    ) -> np.array:
    """ Performs one step of the Forward Euler method
    :param current_state: current state of the system (either height or velocity)
    :param current_height: current height of the system
    :param current_velocity: current velocity of the system 
    :param current_b : current rest height of water
    :param dt: step size of time iteration
    :param dx: step size of space iteration
    :param derivative: function that returns derivatives for current state and time
    :param C: class (struct) with constants
    :return: updated values of dependent variable at next time step
    """
    new_state = current_state + dt * derivative(
        current_height,
        current_velocity,
        current_b,
        dt,
        dx,
        C
    )
    
    return new_state


def leap_frog(# 9 arguments
        prev_state: np.array,
        current_state: np.ndarray,
        current_height: np.ndarray,
        current_velocity: np.ndarray,
        current_b: np.ndarray,
        dt: float,
        dx: float,
        derivative: callable,
        C: Any
    ) -> np.array:
    """ Performs one step of the Leap Frog method

    :param prev_state: previous state of the system (either height or velocity)
    :param current_state: current state of the system (either height or velocity)
    :param current_height: current height of the system
    :param current_velocity: current velocity of the system 
    :param current_b : current rest height of water
    :param dt: step size of time iteration
    :param dx: step size of space iteration
    :param derivative: function that returns derivatives for current state and time
    :param C: class (struct) with constants
    :return: updated values of dependent variable at next time step
    
    """
    new_state = prev_state + 2 * dt * derivative(current_height,
                                              current_velocity,
                                              current_b,
                                              dt, dx, C)

    return new_state


def adam_bashforth(# 10 arguments
        current_state: np.ndarray,
        prev_height: np.ndarray,
        prev_velocity: np.ndarray,
        current_height: np.ndarray,
        current_velocity: np.ndarray,
        current_b: np.ndarray,
        dt: float,
        dx: float,
        derivative: callable,
        C: Any
    ) -> np.array:
    """
    Performs one step of the Adams-Bashforth method
    
    :param prev_height: height of system for the previous timestep
    :param prev_height: velocity of system for the previous timestep
    :param current_state: current state of the system (either height or velocity)
    :param current_height: current height of the system
    :param current_velocity: current velocity of the system 
    :param current_b : current rest height of water
    :param dt: step size of time iteration
    :param dx: step size of space iteration
    :param derivative: function that returns derivatives for current state and time
    :param C: class (struct) with constants
    :return: updated values of dependent variable at next time step
    """
    
    return current_state + dt * \
        (1.5 * derivative(current_height, current_velocity,
                                                  current_b,
                                                  dt, dx, C) \
         - 0.5 * derivative(prev_height, prev_velocity,
                                                   current_b,
                                                   dt, dx, C))


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
    Performs one step of the Crank-nicolson method (with Dirichlet)

    :param current_state: current state of the system (dependent variable)
    :param A: matrix A
    :param B: matrix B
    :param constants: class (struct) with constants
    :return: update to next time step
    """
    intermediate = np.dot(B, current_state)
    intermediate[0], intermediate[-1] = constants.T1, constants.T0
    return np.linalg.solve(A, intermediate)

    
def runge_kutta_4(# 8 arguments
        current_state: np.ndarray,
        current_height: np.ndarray,
        current_velocity: np.ndarray,
        current_b: np.ndarray,
        dt: float,
        dx: float,
        derivative: callable,
        C: Any
    ) -> np.array:
    """ Performs one step of the 4th order Runge-Kutta
    
    :param current_state: current state of the system (either height or velocity)
    :param current_height: current height of the system
    :param current_velocity: current velocity of the system 
    :param current_b : current rest height of water
    :param dt: step size of time iteration
    :param dx: step size of space iteration
    :param derivative: function that returns derivatives for current state and time
    :param C: class (struct) with constants
    :return: updated values of dependent variable at next time step
    """
                            # calculate the 4 k's
    k1 = dt * derivative(current_height,current_velocity, current_b, dt, dx, C)
    k2 = dt * derivative(current_height + 0.5 * dt * k1, current_velocity + 0.5 * dt * k1, current_b, dt, dx, C)
    k3 = dt * derivative(current_height + 0.5 * dt * k2, current_velocity + 0.5 * dt * k2, current_b, dt, dx, C)
    k4 = dt * derivative(current_height + dt    *    k3, current_velocity + dt    *    k3, current_b, dt, dx, C)
                            # calculate and return new state
    return current_state + (k1 + 2 * k2 + 2 * k3 + k4) / 6