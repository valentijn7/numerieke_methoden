# numerieke_methoden/III/task1.py

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


class PhysConstants:
    def __init__(self):
        self.kappa   = 2        # thermal diffusion coefficient (m2/s)
                                #   temperatures:
        self.T0      = 273      # initial temperature rod (K)
        self.T1      = 373      # temperature of rod at x = 0 for t > 0
                                #   rod properties:
        self.L       = 2        # length of the rod (m)
        self.n_x     = 20       # number of gridpoints in the rod
                                # step size of the grid
        self.dx      = self.L / self.n_x
                                #   simulation properties:
        self.t_total = 0.01     # total time of the simulation (s)
        self.n_t     = 1000     # number of time steps
                                # time step of the simulation (s)
        self.dt      = self.t_total / self.n_t


def derivative_heat(
        T: np.array,
        _: float,
        cons: PhysConstants) -> np.array:
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

    return cons.kappa * T_hat


def theoretical(
        x_arr: np.array,
        t: int,
        C: PhysConstants
    ) -> np.array:
    """ Theoretical solution of the heat equation;
    the erfc function is imported from the scipy library:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfc.html

    :param x_arr: array with grid positions
    :param t: current time
    :return: array with temperatures at grid positions
    """
    if t == 0:
        return np.full_like(x_arr, C.T0)
    else:
        return C.T0 + (C.T1 - C.T0) * special.erfc(x_arr / (2 * np.sqrt(C.kappa * t)))


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
    C.n_x = int(nx)
    C.dx = L / nx
    C.dt = dt
    C.n_t = int(C.t_total / dt)
                                # first, we define a grid, or matrix, for space and time;
    grid = np.ndarray(shape = (nx + 1, C.n_t + 1), dtype = float)
    grid[:] = np.nan
    grid[:, 0] = C.T0
    grid[0, :] = C.T1
    grid[-1, :] = C.T0
                                # second, we define the space and time arrays
    time = np.linspace(0, t_total, C.n_t + 1)
    space = np.linspace(0, L, nx + 1)
                                # third, solve heat equation
    if TimeSteppingMethod == "Theory":
        for idx in range(1, len(time)):
            grid[:, idx] = theoretical(space, time[idx], C)

    elif TimeSteppingMethod == "FE":
        for idx in range(1, len(time)):
            grid[:, idx] = forward_euler(grid[:, idx - 1], time[idx - 1], dt, derivative_heat, C)

    elif TimeSteppingMethod == "LF":
        for idx in range(1, len(time)):
            if idx == 1:
                grid[:, idx] = forward_euler(grid[:, idx - 1], time[idx - 1], dt, derivative_heat, C)
            else:
                grid[:, idx] = leap_frog(grid[:, idx - 2], grid[:, idx - 1], time[idx - 1], dt, derivative_heat, C)

    elif TimeSteppingMethod == "AB":
        for idx in range(1, len(time)):
            if idx == 1:
                grid[:, idx] = forward_euler(grid[:, idx - 1], time[idx - 1], dt, derivative_heat, C)
            else:
                grid[:, idx] = adam_bashforth(grid[:, idx - 2], grid[:, idx - 1], time[idx - 1], dt, derivative_heat, C)

    elif TimeSteppingMethod == "CN":
        A, B = init_crank_nicolson(C)
        for idx in range(1, len(time)):
            grid[:, idx] = crank_nicolson(grid[:, idx - 1], A, B, C)

    elif TimeSteppingMethod == "RK4":
        for idx in range(1, len(time)):
            grid[:, idx] = runge_kutta_4(grid[:, idx - 1], time[idx - 1], dt, derivative_heat, C)

    else:
        raise ValueError("Invalid TimeSteppingMethod")

    return time, space, grid


def MSE(
        gt: np.array,
        pred: np.array
    ) -> float:
    """ Calculate the Mean Squared Error between two arrays

    :param gt: ground truth array
    :param pred: predicted array
    :return: RMSE value
)
    """
    return np.mean((gt - pred)**2)


def MSE_per_t(
        gt: np.array,
        pred: np.array
    ) -> np.array:
    """ Calculate the Mean Squared Error between two grids at each time step
)
    :param gt: ground truth grid
    :param pred: predicted grid
    :return: RMSE values at each time step
    """
    return np.mean((gt - pred)**2, axis = 0)


def find_optimal_sigmas(
        C: PhysConstants
    ) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Find the optimal sigmas (in pseudo-pseudocode):
    - for each method and kappa:
        - try dx candidates:
            - try dt candidates:
            - if it blows up, store previous dt
        - choose the dx with the highest succesful dt
        - store sigma, dx, dt, kappa in dictionary
    - return dictionary

    In other words, it searches a point within 3D search space

    :param C: class (struct) with constants
    :return: dictionary with optimal sigma for each method
    """
    params = {}
    methods = ["FE", "LF", "AB", "CN", "RK4"]
    dx_candidates = np.logspace(-2, 0, 30)
    dt_candidates = np.logspace(-5, 10, 16)
    kappa_candidates = np.linspace(1.1, 4, 5)

    for method in methods:
        if method == "CN": #! TODO verwijder dit na toevoegen CN
            params[method] = (None, None, None)
            continue

        print(f'Searching params for {method}...')
        final_kappa_dxs = np.zeros(len(kappa_candidates))
        final_kappa_dts = np.zeros(len(kappa_candidates))
        max_dts = np.zeros(len(dx_candidates))

        for idx_kappa, kappa in enumerate(kappa_candidates):
            print(f'Searching for kappa = {kappa}')
            C.kappa = kappa

            for idx_dx, dx in enumerate(dx_candidates):
                print(f'\tSearching for max stable dt with dx = {dx}...')
                nx = int(C.L / dx) + 1

                for idx_dt, dt in enumerate(dt_candidates):
                    _, _, grid = Task1_caller(C.L, nx, C.t_total, dt, method)
                    if np.isnan(grid).any() or np.isinf(grid).any() or np.max(grid) > C.T1: 
                        max_dts[idx_dx] = dt_candidates[idx_dt - 1]
                        break
        
            max_dx = dx_candidates[np.argmax(max_dts)]
            max_dt = np.max(max_dts)
            final_kappa_dxs[idx_kappa] = max_dx
            final_kappa_dts[idx_kappa] = max_dt
        
        final_dt = np.max(final_kappa_dts)
        final_dx = final_kappa_dxs[np.argmax(final_kappa_dts)]
        
        sigma = (C.kappa * final_dt) / final_dx**2
        params[method] = (sigma, final_dx, final_dt, kappa)
        print('\n\n\n')

    return params


def main():
    start_time = datetime.now()
    C = PhysConstants()

    # we search for optimal sigma's first, i.e. sigma's that are as big as possible
    # while still stable. we do this with a fixed dx, and iterate dts till it blows up.
    # This process is embedded into the calls to Task1_caller below, and the found dx's and dt's
    # are used in the final simulation and also printed for reference
    # params = find_optimal_sigmas(C)
    # params is a dictionary with method as key, values are sigma, dx, dt as tuple
    # print(params)

    """

    """
    
 


    time, space, grid_th = Task1_caller(
        C.L,
        C.n_x,
        C.t_total,
        C.dt,
        "Theory")

    time, space, grid_FE = Task1_caller(
        C.L,
        C.n_x,
        C.t_total,
        C.dt,
        "FE")
    
    time, space, grid_LF = Task1_caller(
        C.L,
        C.n_x,
        C.t_total,
        C.dt,
        "LF")
    
    time, space, grid_AB = Task1_caller(
        C.L,
        C.n_x,
        C.t_total,
        C.dt,
        "AB")
    
    time, space, grid_CN = Task1_caller(
        C.L,
        C.n_x,
        C.t_total,
        C.dt,
        "CN")
    
    time, space, grid_RK4 = Task1_caller(
        C.L,
        C.n_x,
        C.t_total,
        C.dt,
        "RK4")
    
    ########## plotting ##########

    grids = [grid_th, grid_FE, grid_LF, grid_AB, grid_CN, grid_RK4]
    markers = ['o', 's', '^', 'D', 'v', 'x']
    marker_ints = [2, 3, 5, 7, 11]

    fig, ax = plt.subplots()    # unpack Tuple with , for single element
    line_th, = ax.plot([], [], lw = 2, label = "theoretical")
    line_FE, = ax.plot([], [], lw = 2, label = "forward Euler", marker = markers[0], markevery = marker_ints[0])
    line_LF, = ax.plot([], [], lw = 2, label = "leap-frog", marker = markers[1], markevery = marker_ints[1])
    line_AB, = ax.plot([], [], lw = 2, label = "Adams-Bashforth", marker = markers[2], markevery = marker_ints[2])
    line_CN, = ax.plot([], [], lw = 2, label = "Crank-nicolson", marker = markers[3], markevery = marker_ints[3])
    line_RK4, = ax.plot([], [], lw = 2, label = "Runge-Kutta 4", marker = markers[4], markevery = marker_ints[4])

                                # set limits, labels, grid, legend, ticks
    ax.set_xlim(space[0], space[-1])
    ax.set_ylim(np.nanmin(grids), np.nanmax(grids))
    ax.set_xlabel("rod (m)", fontsize = 20)
    ax.set_ylabel("temperature (K)", fontsize = 20)
    ax.grid(True)
    plt.legend(fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)

    def init():                 # create empty plot animation
        line_FE.set_data([], [])
        line_LF.set_data([], [])
        line_AB.set_data([], [])
        line_CN.set_data([], [])
        line_RK4.set_data([], [])
        line_th.set_data([], [])
        return line_FE, line_LF, line_AB, line_RK4, line_th

    def update(frame):          # update plot animation with new data
                                # theoretical solution
        y_th = grid_th[:, frame]
        line_th.set_data(space, y_th)
                                # numerical solutions
        y_FE = grid_FE[:, frame]
        line_FE.set_data(space, y_FE)
        y_LF = grid_LF[:, frame]
        line_LF.set_data(space, y_LF)
        y_AB = grid_AB[:, frame]
        line_AB.set_data(space, y_AB)
        y_CN = grid_CN[:, frame]
        line_CN.set_data(space, y_CN)
        y_RK4 = grid_RK4[:, frame]
        line_RK4.set_data(space, y_RK4)

        ax.set_title(f"t = {time[frame]:.4f} s", fontsize = 20)
        return line_FE, line_LF, line_AB, line_CN, line_RK4, line_th

    # https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
    ani = animation.FuncAnimation(fig,
                                  update,
                                  frames = len(time),
                                  init_func = init,
                                  interval = 100)
    plt.show()


    # Next, we plot our data in stationary graphs. We get 5 plots, one for each method's
    # comparison with the theoretical solution.
    d_results = {'Forward Euler' : grid_FE,
                 'Leap-frog' : grid_LF,
                 'Adams-Bashforth' : grid_AB,
                 'Crank-nicolson' : grid_CN,
                 'Runge-Kutta 4' : grid_RK4
                 }
    colours = ('#0A97B0', '#AE445A', '#2A3335', '#F29F58',
               '#355F2E', '#AE445A', '#A27B5C', '#FFCFEF')
    

    for method in d_results.keys():
        plt.figure(figsize = (15, 4))

        for t, col in zip(range(0, C.n_t, int(C.n_t / 5)), colours):
            plt.plot(space, d_results[method][:, t], marker = '.', color = col,
                     label = f't = {time[t]:.4f}s')
            plt.plot(space, grid_th[:, t], marker = '.', color = col, linestyle = ':')
        plt.title(f'{method} vs. theory (dashed)', fontsize = 20)
        plt.xlabel('Position on rod $(m)$', fontsize = 20)
        plt.ylabel('Temperature $(K)$', fontsize = 20)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.legend(fontsize = 20)
        plt.grid()
        plt.show()


    # Next, we plot the error of each method compared to the theoretical solution
    # in one combined plot. Error calculated as Mean Squared Error
    d_errors = {}
    for method in d_results.keys():
        d_errors[method] = MSE_per_t(grid_th, d_results[method])

    for idx, method in enumerate(d_errors.keys()):
        plt.plot(time, d_errors[method], color = colours[idx], label = method,
                 marker = markers[idx % len(markers)], markevery = marker_ints[idx % len(marker_ints)])
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which = 'both')
    plt.title('method vs. theory: MSE', fontsize = 20)
    plt.xlabel('log(time $(s))$', fontsize = 20)
    plt.ylabel('log(MSE $(T^2)$)', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(fontsize = 20)
    plt.show()


    print(f"\nexecution time: {datetime.now() - start_time}")


if __name__ == "__main__":
    main()