# numerieke_methoden/III/Task1_MandatoryStructure.py

"""
@author: Maarten van der Molen & Valentijn Oldenburg
"""
from typing import List, Dict, Tuple, Any, Callable
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class PhysConstants:
    def __init__(self):
        self.kappa   = 2        # thermal diffusion coefficient (m2/s)
                                #   temperatures:
        self.T0      = 273      # initial temperature rod (K)
        self.T1      = 373      # temperature of rod at x = 0 for t > 0
                                #   rod properties:
        self.L       = 10       # length of the rod (m)
        self.n_x     = 100       # number of gridpoints in the rod
                                # step size of the grid
        self.dx      = self.L / self.n_x
                                #   simulation properties:
        self.t_total = 0.05     # total time of the simulation (s)
        self.dt      = 0.001        # time step of the simulation (s)
                                # number of time steps
        self.n_t     = int(self.t_total / self.dt)


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


def RK4(
        current_state: np.array,
        current_time: float,
        step_size: float,
        derivative: callable,
        constants: PhysConstants
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


def derivative_heat(
        T: np.array,
        _: float,
        cons: PhysConstants) -> float:
    """ Calculates the derivative of the rod using the heat equation
    
    :param T: temperature array of the rod
    :param _: we discard the time parameter
    :param cons: class (struct) with constants
    :return: central derivative of the array at the given index
    """
    T_hat = T.copy()

    T_hat[0] = 0.0          # At boundaries temperature is fixed, aka derivative = 0
    T_hat[-1] = 0.0
    
    for idx in range(1, len(T) - 1):
        T_hat[idx] = (T[idx + 1] - 2.0 * T[idx] + T[idx - 1]) / (cons.dx**2)

    return T_hat


def double_derivative_heat(
        T: np.array,
        _: float,
        cons: PhysConstants
    ) -> float:
    """ Wrapper function for double derivative

    :param T: temperature array of the rod
    :param _: we discard the time parameter
    :param cons: class (struct) with constants
    :return: second derivative of the array
    """
    return cons.kappa * derivative_heat(T, _, cons)


def Task1_caller(L: int,
                 nx: int,
                 t_total: int,
                 dt: int,
                 TimeSteppingMethod: str, 
                 DiffMethod: str = "CD"
    ) -> Tuple[np.array, np.array, np.array]:
    """ This routine calls the function that solves the heat equation

    The mandatory input is:
    L                   Length of domain to be modelled (m)
    nx                  Number of gridpoint in the model domain
    TotalTime           Total length of the simulation (s)
    dt                  Length of each time step (s)
    TimeSteppingMethod  Could be:
     "Theory"             Theoretical solution
     "AB"                 Adams-Bashforth
     "CN"                 Crank-Nicholson
     "EF"                 Euler Forward
     "LF"                 Leaf Frog
     "RK4"                Runge-Kutta 4
    
    The optional input is:
    DiffMethod  Method to determine the 2nd order spatial derivative
      Default = "CD"    Central differences
       Option = "PS"    Pseudo spectral
    
    The output is:
    Time        a 1-D array (length nt) with time values considered
    Xaxis       a 1-D array (length nx) with x-values used
    Result      a 2-D array (size [nx, nt]), with the results of the routine    
    """
    C = PhysConstants() 
                                # first, we define a grid, or matrix, for space and time;
    grid = np.ndarray(shape = (C.n_x, C.n_t), dtype = float)
    grid[:] = np.nan
    grid[:, 0] = C.T0
    grid[0, :] = C.T1
    grid[:, -1] = 0.0 
                                # second, we define the space and time arrays
    time = np.arange(0, t_total, dt)
    space = np.arange(0, L, L / nx)
                                # third, solve heat equation
    for idx in range(1, len(time)):
        grid[:, idx] = RK4(grid[:, idx - 1], time[idx - 1], dt, double_derivative_heat, C)

    return time, space, grid   


def theoretical(
        x_arr: np.array,
        t: int,
        C: PhysConstants
    ) -> np.array:
    """ Theoretical solution of the heat equation.
    The erfc function is imported from the scipy library:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfc.html

    :param x_arr: array with grid positions
    :param t: current time
    :return: array with temperatures at grid positions
    """
    if t == 0:
        return np.full_like(x_arr, C.T0)
    else:
        return C.T0 + (C.T1 - C.T0) * special.erfc(x_arr / (2 * np.sqrt(C.kappa * t)))


def analyse_results(
        time: np.array,
        space: np.array,
        grid: np.array
    ) -> None:
    """ Analyse the results with e.g. plots

    :param time: time array
    :param space: space array
    :param grid: grid with results
    """
    fig, ax = plt.subplots()    # unpack Tuple with , for single element
    line_th, = ax.plot([], [], lw = 2, color = 'blue')
    line_num, = ax.plot([], [], lw = 2, color = 'red')

                                # print the grid for debugging purposes
    for idx in range(len(grid[:, ])):
        print(grid[idx, :])
                                # set limits, labels, grid, legend, ticks
    ax.set_xlim(space[0], space[-1])
    ax.set_ylim(np.nanmin(grid), np.nanmin(grid))
    ax.set_xlabel("rod (m)", fontsize = 16)
    ax.set_ylabel("temperature (K)", fontsize = 16)
    ax.grid(True)
    ax.legend(["Theoretical", "Numerical"], fontsize = 16)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)

    def init():                 # create empty plot animation
        line_num.set_data([], [])
        line_th.set_data([], [])
        return line_num, line_th

    def update(frame):          # update plot animation with new data
        y_num = grid[:, frame]  # numerical solution
        line_num.set_data(space, y_num)
                                # theoretical solution
        y_th = theoretical(space, time[frame], PhysConstants())
        line_th.set_data(space, y_th)

        ax.set_title(f"t = {time[frame]:.2f} s", fontsize = 16)
        return line_num, line_th

    # https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
    ani = animation.FuncAnimation(fig,
                                  update ,
                                  frames = len(time),
                                  init_func = init,
                                  interval = 100)

    plt.show()


def main():
    C = PhysConstants()

    time, space, grid = Task1_caller(
        C.L,
        C.n_x,
        C.t_total,
        C.dt,
        "RK4")
    
    analyse_results(time, space, grid)


if __name__ == "__main__":
    main()