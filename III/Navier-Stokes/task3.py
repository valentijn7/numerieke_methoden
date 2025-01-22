# numerieke_methoden/III/Navier-Stokes/task3.py

from typing import List, Dict, Tuple, Any, Callable
from scipy import special
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from algorithms import forward_euler
from algorithms import leap_frog
from algorithms import adam_bashforth
from algorithms import init_crank_nicolson
from algorithms import crank_nicolson
from algorithms import runge_kutta_4


# Note that every time the term "velocity" is used in this script,
# what is meant is horizontal velocity.


class PhysConstants:
    def __init__(self):
        self.g       = 9.81     # gravitational acceleration
        self.b       = 1        # height of water at rest (constant)
        self.beta    = 0.01     # the constant parametrising friction

                                #   water column properties:
        self.L       = 0.5      # horizontal length of the column (m)
        self.n_x     = 20       # number of gridpoints in the column
                                # step size of the grid
        self.dx      = self.L / self.n_x

                                #   simulation properties:
        self.t_total = 0.01     # total time of the simulation (s)
        self.n_t     = 1000     # number of time steps
                                # time step of the simulation (s)
        self.dt      = self.t_total / self.n_t


def central_derivative(
        state: np.ndarray, 
        step_size: float
    ) -> np.array:
    """
    Uses central differentiation to differentiate a given input array.
    At the ends, forward and backward differentiation are used.
    
    :param state: input array of the variable
    :param step_size: size of the step over which is differentiated.
    :return: differentiated array
    """
    central_diff = np.zeros(len(state))
    central_diff[0] = (state[1] - state[0]) / step_size
    central_diff[-1] = (state[-1] - state[-2]) / step_size
    central_diff[1:-1] = (state[2: ] - state[ :-2]) / (2 * step_size)
    
    return central_diff


def H_time_deriv(
        height: np.ndarray, 
        velocity: np.ndarray,
        b: np.ndarray,
        dt: float,
        dx: float,
        C: Any
    ) -> np.ndarray:
    """
    Function that generates values for the time derivative of the height
    for one iteration step over dt

    :param height: array containing the heights of the water at point t
    :param velocity: array containing the (horizontal) velocities of the water at point t
    :param b: array of water heights at rest
    :param dt: temporal step size
    :param dx: spacial step size
    :param C: class containing physical constants
    :return: array containing the time derivative of the height
    """
                        # first:  create array for H*u;
                        # second: use CD to get the spacial derivative of H*u,
                        #         and compute dH_dt
    H_u = height * velocity
    dH_dt = -central_derivative(H_u, dx)
    return dH_dt


# def u_time_deriv(height: np.ndarray, 
#                  velocity: np.ndarray,
#                  b: np.ndarray,
#                  dt: float,
#                  dx: float,
#                  C: Any
#                  ) -> np.ndarray:
#     """ 
#     Function that generates values for the time derivative of the velocity
#     for one iteration step over dt
    
#     :param height: array containing the heights of the water at point t
#     :param velocity: array containing the (horizontal) velocities of the water at point t
#     :param b: array of water heights at rest
#     :param dt: temporal step size
#     :param dx: spacial step size
#     :param C: class containing physical constants
#     :return: array containing the time derivative of the velocity
#     """
#                         # first:  calculate dH_dt;
#                         # second: create an array for H*u^2;
#                         # third:  use CD to get the spacial derivative of H*u^2;
#                         # fourth: calculate the third term of the equation;
#                         # fifth:  calculate the time derivative of the velocity
#     dH_dt = H_time_deriv(height, velocity, b, dt, dx, C)
#     H_u2 = height * velocity * velocity
#     dzeta_dx =  central_derivative(height, dx)
#     third_term = C.g * height * dzeta_dx
#     return (-velocity * dH_dt - central_derivative(H_u2, dx) - third_term - C.beta * velocity / height) / height


def Hu_time_deriv(
        height: np.ndarray, 
        velocity: np.ndarray,
        b: np.ndarray,
        dt: float,
        dx: float,
        C: Any
    ) -> np.ndarray:
    """ 
    Function that generates values for the time derivative
    of the velocity for one iteration step over dt
    
    :param height: array containing the heights of the water at point t
    :param velocity: array containing the (horizontal) velocities of the water at point t
    :param b: array of water heights at rest
    :param dt: temporal step size
    :param dx: spacial step size
    :param C: class containing physical constants
    :return: array containing the time derivative of the velocity
    """                     
    H_u2 = height * velocity**2
    helper_var = H_u2 + (C.g / 2) * height**2
    return -central_derivative(helper_var, dx) - C.beta * (velocity / height)


def Task3_caller(
        L: int,
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
     "AB"                 Adams-Bashforth
     "CN"                 Crank-nicolson
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
    C = PhysConstants()         # init PhysConstants and recalculate based on input params
    C.L = L
    C.n_x = int(nx)
    C.dx = L / nx
    C.t_total = t_total
    C.dt = dt
    C.n_t = int(C.t_total / dt)
                                
                                # init grids for H and Hu
    h_grid = np.zeros((nx + 1, C.n_t + 1), dtype = float)
    Hu_grid = np.zeros((nx + 1, C.n_t + 1), dtype = float)
                                # for h, set first row to 1, and first column to 2
    h_grid[:, 0] = 1.0
    h_grid[0, :] = 2.0
                                # init u grid with 1, and set first row to 3
    u_grid = np.ones((nx + 1, C.n_t + 1))
    u_grid[0, :] = 3.0
                                # fill Hu grid with product of h and u
    Hu_grid = u_grid * h_grid
                                # b array is filled with constant value
    b_arr = np.zeros(nx + 1)
    b_arr[:] = C.b
                                # create time and space arrays
    time = np.linspace(0, t_total, C.n_t + 1)
    space = np.linspace(0, L, nx + 1)

    #TODO! Implementeer realistische golven! 
    #TODO! Check welke methode echt klopt, de u- of de Hu-methode!!!
                                
    if TimeSteppingMethod == "FE":
        for idx in range(1, len(time)):
            H_old = h_grid[:, idx - 1]
            Hu_old = Hu_grid[:, idx - 1]
            u_old = Hu_old / H_old

            h_grid[:, idx] = forward_euler(
                H_old,
                H_old, 
                u_old,
                b_arr,
                C.dt,
                C.dx,
                H_time_deriv,
                C
            )
            Hu_grid[:, idx] = forward_euler(
                Hu_old,
                H_old,
                u_old,
                b_arr,
                C.dt,
                C.dx,
                Hu_time_deriv,
                C
            )

    # elif TimeSteppingMethod == "LF":
    #     for idx in range(1, len(time)):
    #         if idx == 1:
    #             h_grid[:, idx] = forward_euler(h_grid[:, idx - 1], h_grid[:, idx - 1],
    #                                            Hu_grid[:, idx - 1], b_arr,
    #                                            C.dt, C.dx, H_time_deriv, C)
    #             Hu_grid[:, idx] = forward_euler(Hu_grid[:, idx - 1], h_grid[:, idx - 1],
    #                                            Hu_grid[:, idx - 1], b_arr,
    #                                            C.dt, C.dx, Hu_time_deriv, C)
    #         else:
    #             h_grid[:, idx] = leap_frog(h_grid[:,idx - 2], h_grid[:, idx - 1], h_grid[:, idx - 1],
    #                                            Hu_grid[:, idx - 1], b_arr,
    #                                            C.dt, C.dx, H_time_deriv, C)
    #             Hu_grid[:, idx] = leap_frog(Hu_grid[:, idx - 2], Hu_grid[:, idx - 1], h_grid[:, idx - 1],
    #                                            Hu_grid[:, idx - 1], b_arr,
    #                                            C.dt, C.dx, Hu_time_deriv, C)

    # elif TimeSteppingMethod == "AB":
    #     for idx in range(1, len(time)):
    #         if idx == 1:
    #             h_grid[:, idx] = forward_euler(h_grid[:, idx - 1], h_grid[:, idx - 1],
    #                                            Hu_grid[:, idx - 1], b_arr,
    #                                            C.dt, C.dx, H_time_deriv, C)
    #             Hu_grid[:, idx] = forward_euler(Hu_grid[:, idx - 1], h_grid[:, idx - 1],
    #                                            Hu_grid[:, idx - 1], b_arr,
    #                                            C.dt, C.dx, Hu_time_deriv, C)
    #         else:
    #             h_grid[:, idx] = adam_bashforth(h_grid[:, idx - 1], h_grid[:, idx - 2], 
    #                                            Hu_grid[:, idx - 2], h_grid[:, idx - 1],
    #                                            Hu_grid[:, idx - 1], b_arr,
    #                                            C.dt, C.dx, H_time_deriv, C)
    #             Hu_grid[:, idx] = adam_bashforth(Hu_grid[:, idx - 1], h_grid[:, idx - 2], 
    #                                            Hu_grid[:, idx - 2], h_grid[:, idx - 1],
    #                                            Hu_grid[:, idx - 1], b_arr,
    #                                            C.dt, C.dx, Hu_time_deriv, C)

    # elif TimeSteppingMethod == "CN":
    #     h_grid[:] = 1
    #     Hu_grid[:] = 1
    #     # A, B = init_crank_nicolson(C)
    #     # for idx in range(1, len(time)):
    #     #     grid[:, idx] = crank_nicolson(grid[:, idx - 1], A, B, C)
    #     # TODO! maak Crank nicolson methode voor deze -> A,B maken
        
    # elif TimeSteppingMethod == "RK4":
    #     for idx in range(1, len(time)):
    #             h_grid[:, idx] = runge_kutta_4(h_grid[:, idx - 1], h_grid[:, idx - 1],
    #                                            Hu_grid[:, idx - 1], b_arr,
    #                                            C.dt, C.dx, H_time_deriv, C)
    #             Hu_grid[:, idx] = runge_kutta_4(Hu_grid[:, idx - 1], h_grid[:, idx - 1],
    #                                            Hu_grid[:, idx - 1], b_arr,
    #                                            C.dt, C.dx, Hu_time_deriv, C)
    else:
        raise ValueError("Invalid TimeSteppingMethod")
    
    u_grid_final = Hu_grid / h_grid

    return time, space, (h_grid, u_grid_final)


def main():
    C = PhysConstants()
    C.L = 0.5
    C.n_x = 20
    C.t_total = 0.01
    C.dt = 1e-5

    time, space, FE_grid_tuple = Task3_caller(
        C.L,
        C.n_x,
        C.t_total,
        C.dt,
        "FE")
    h_grid, u_grid_final = FE_grid_tuple

    indices = np.linspace(0, len(time) - 1, 5, dtype = int)
    plt.figure(figsize = (10, 5))
    
    for idx in indices:
        plt.plot(space, h_grid[:, idx], marker = 'o', label = f't = {time[idx]:.6f}s')
    
    plt.title('column height over position for different t (FE)', fontsize = 20)
    plt.xlabel('position x (m)', fontsize = 20)
    plt.ylabel('height H (m)', fontsize = 20)
    plt.legend(fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # time, space, LF_grid_tuple = Task3_caller(
    #     C.L,
    #     C.n_x,
    #     C.t_total,
    #     C.dt,
    #     "LF")
    
    # time, space, AB_grid_tuple = Task3_caller(
    #     C.L,
    #     C.n_x,
    #     C.t_total,
    #     C.dt,
    #     "AB")
    
    # time, space, CN_grid_tuple = Task3_caller(
    #     C.L,
    #     C.n_x,
    #     C.t_total,
    #     C.dt,
    #     "CN")
    
    # time, space, RK4_grid_tuple = Task3_caller(
    #     C.L,
    #     C.n_x,
    #     C.t_total,
    #     C.dt,
    #     "RK4")


    # height_grids = [FE_grid_tuple[0], LF_grid_tuple[0], AB_grid_tuple[0], 
    #                 CN_grid_tuple[0], RK4_grid_tuple[0]]
    # velocity_grids = [FE_grid_tuple[1], LF_grid_tuple[1], AB_grid_tuple[1], 
    #                 CN_grid_tuple[1], RK4_grid_tuple[1]]
    
    # # Next, we plot our data in stationary graphs. We get 5 plots, one for each method's
    # # comparison with the theoretical solution.
    # d_h_results = {'Forward Euler' : FE_grid_tuple[0],
    #              'Leap-frog' : LF_grid_tuple[0],
    #              'Adams-Bashforth' : AB_grid_tuple[0],
    #              'Crank-nicolson' : CN_grid_tuple[0],
    #              'Runge-Kutta 4' : RK4_grid_tuple[0]
    #              }
    
    # d_v_results = {'Forward Euler' : FE_grid_tuple[1],
    #              'Leap-frog' : LF_grid_tuple[1],
    #              'Adams-Bashforth' : AB_grid_tuple[1],
    #              'Crank-nicolson' : CN_grid_tuple[1],
    #              'Runge-Kutta 4' : RK4_grid_tuple[1]
    #              }
    # d_variables = {'Height': d_h_results, 'Velocity': d_v_results}
    
    # colours = ('#0A97B0', '#AE445A', '#2A3335', '#F29F58',
    #            '#355F2E', '#AE445A', '#A27B5C', '#FFCFEF')
    
    # for name, variables in d_variables.items():
    #     plt.figure()
    #     plt.plot(1,1) # Dit is voor de duidelijkheid, later weghalen!
    #     plt.show
    #     for method in variables.keys():
    #         plt.figure(figsize = (15, 4))
    
    #         for t, col in zip(range(0, C.n_t, int(C.n_t / 5)), colours):
    #             plt.plot(space, variables[method][:, t], marker = '.', color = col,
    #                      label = f't = {time[t]:.4f}s')
    #         plt.title(f'{name} of fluid computed through {method}', fontsize = 20)
    #         plt.xlabel('Position from end of basin $(m)$', fontsize = 20)
    #         plt.ylabel('Height from bottom of basin $(m)$', fontsize = 20)
    #         plt.xticks(fontsize = 20)
    #         plt.yticks(fontsize = 20)
    #         plt.legend(fontsize = 20, loc = 'upper right')
    #         plt.grid()
    #     plt.show()


if __name__ == '__main__':
    main()