# numerieke_methoden/III/advection/task2.py

from typing import List, Dict, Tuple, Any, Callable
from scipy import special
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from algorithms_task2 import forward_euler
from algorithms_task2 import leap_frog
from algorithms_task2 import adam_bashforth
from algorithms_task2 import init_crank_nicolson
from algorithms_task2 import crank_nicolson
from algorithms_task2 import runge_kutta_4


class PhysConstants:
    def __init__(self):
        self.c       =    2         # advection velocity (m/s)
        self.A_g     =    50        # Gaussian wave amplitude ()
        self.sig_g   =    0.8       # Gaussian wave width (m)
        self.A_0     =    10        # Molenkamp triangle height ()
        self.W       =    3         # Molenkamp triangle width (m)
        
        self.L       =    10        # length of domain (m)
        self.n_x     =    200       # amount of gridpoints
                                    # spatial gridstep size
        self.dx      =    self.L / self.n_x
        
        self.t_total =    30        # total time of simulation (s)
        self.sigma   =    1  
        self.dt      =    self.dx * self.sigma / self.c
        self.n_t     =    self.t_total / self.dt


 
def CD(
        state: np.ndarray, 
        step_size: float
    ) -> np.array:
    """
    Uses central differentiation to differentiate a given input array;
    at the ends, forward and backward differentiation are used
    
    :param state: input array of the variable
    :param step_size: size of the step over which is differentiated, spatial in our case
    :return: differentiated array
    """
    central_diff = np.zeros(len(state))
    central_diff[0] = (state[1] - state[0]) / step_size
    central_diff[-1] = (state[-1] - state[-2]) / step_size
    central_diff[1:-1] = (state[2: ] - state[ :-2]) / (2 * step_size)
    
    return central_diff


def CD_periodic(
        state: np.ndarray, 
        step_size: float
    ) -> np.ndarray:
    """ Uses central differentiation to differentiate a given input array,
    however, now it is done periodically: i.e. the index before the 0th
    index is the -1th index and so on
    
    :param state: input array of variable
    :param step_size: size of the step over which is differentiated
    :return: differentiated array
    """
    # np.roll makes indexing periodically a bit easier (though it's not strictly necessary)
    state_right = np.roll(state, -1)
    state_left  = np.roll(state,  1)
    
    return (state_right - state_left) / (2.0 * step_size)


def derivatives(
        current_state: np.ndarray,
        current_time: float,
        C: Any
    ) -> np.ndarray:
    """ Generates the time derivative of u from the advection PDE.
    For the  spatial derivative of u, CD is used.
    
    :param current_state: input array of variable with current values
    :param current_time: current time
    :return: array of du_dt-values for every argument of the current_state
    """
    return - C.c * CD_periodic(current_state, C.dx)


def derivatives_PS(
        current_state: np.ndarray,
        current_time: float,
        C: Any
    ) -> np.ndarray:
    """ Computes pseudo-spectral derivatives of the current state;
    again, current_time is unused. We used the lecture slides as inspiration

    :param current_state: input array of variable with current values
    :param current_time: current time
    :return: array of du_dt-values for every argument of the current_state
    """
                                # first transform the current state to Fourier space       
    state_hat = np.fft.rfft(current_state)
                                # then multiply with the Fourier space of the derivative
    indices = np.arange(state_hat.shape[0])
    dudx_hat = 1j * (indices * 2 * np.pi / C.L) * state_hat
                                # and transform back to real space, multiplying with -c
    return -C.c * np.fft.irfft(dudx_hat, n = len(current_state))


def init_gauss_wave(
        space: np.ndarray,
        midpoint: float,
        C: Any,
        current_time: float = 0,
    ) -> np.ndarray:
    """  Creates an array with a Gaussian shape
        
    :param space: spatial grid
    :param midpoint: midpoint of the Gaussian wave (though it can
                     also be set more to the left or right)
    :param C: class (struct) with constants
    :param current_time: current time of the simulation (default = 0)
    :return: init'ed array with Gaussian values
    """
    return C.A_g * np.e **(-((-C.c * current_time + space - midpoint)**2) / C.sig_g**2)
    

def init_molenkamp_wave(
        space: np.ndarray,
        midpoint: float,
        C: Any
    ) -> np.ndarray:
    """ Creates an array with values that decribe the Molenkamp wave
    
    :param space: spatial grid
    :param midpoint: midpoint of the Molenkamp wave
    :param C: class (struct) with constants
    :return: init'ed array with Molenkamp values
    """                     
                            # make a copy to avoid memory issues
    molen_arr = space.copy()
                            # determine boundary indices of rising and
                            # falling part of the Molenkamp wave
                            # and subsequently fill the array with the
                            # corresponding values; the rest is zero
    max_idx_1 = np.argmax(space >= C.W / 2)
    max_idx_2 = np.argmax(space >= C.W)

    molen_arr[0: max_idx_1 + 1] = \
        2 * C.A_0 * space[0: max_idx_1 + 1] / C.W
    molen_arr[max_idx_1 + 1: max_idx_2 + 1] = \
        2 * C.A_0 * (C.W - space[max_idx_1 + 1: max_idx_2 + 1]) / C.W
    molen_arr[max_idx_2 + 1:] = 0

    return molen_arr


def theoretical_gauss(
        space: np.ndarray,
        current_time: float,
        midpoint: float,
        C: Any
    ) -> np.ndarray:
    """ 
    Gives the theoretical values of the Gaussian wave at a certain time
    (similar to the init_gauss_wave function, but now periodic as well)
        
    :param space: spatial grid
    :param current_time: current time
    :param midpoint: center of the Gaussian wave at (t = 0)
    :param C: class (struct) with constants
    :return: Array with the Gaussian values
    """
    # To prevent possible messing with the original value
    space_copy = space.copy()
    # In order to force the wave to be periodic, the spatial argument
    # of our exponent cannot exceed C.L
    shifted_space = np.mod(space_copy - C.c * current_time - midpoint, C.L)
    # We also need to make sure the wave is symmetric around our point
    # of symmetry C.L / 2.
    shifted_space = np.where(shifted_space > C.L / 2, shifted_space - C.L, shifted_space)
    # Fill in our arguments
    gauss_arr = C.A_g * np.exp(-shifted_space**2 / C.sig_g**2)
    return gauss_arr



def theoretical_molenkamp(
        space: np.ndarray,
        current_time: float,
        C: Any
    ) -> np.ndarray:
    """ Theoretical solution for Molenkamp wave; similar to
    the init_molenkamp_wave function

    The wave is shifted (moved) based on velocity c and time current_time,
    with periodic boundary conditions in a domain of length L.

    :param space: 1D numpy array (spatial grid points from 0 to L, say).
    :param current_time: The current time (used to shift the wave).
    :param C: A constants/config object with at least:
              - C.W   (float): width of the Molenkamp wave
              - C.A_0 (float): amplitude scaling factor
              - C.c   (float): wave speed
              - C.L   (float): length of the domain (for periodic wrapping)
    :return: 1D numpy array (same shape as space) with wave values at current_time.
    """

    # Initialize output array to zeros.
    molen_arr = np.zeros_like(space)

    # Calculate wave shift dependent on the current_time
    shift = C.c * current_time
    
    # Apply the shift and wrap around the domain [0, L).
    #    This effectively puts the wave's "reference" at x - c*t (mod L).
    #    Now 'shifted_space[i]' is the coordinate in the wave’s local frame.
    shifted_space = (space - shift) % C.L  # shape = same as space

    # Define the wave shape *in the local frame*, i.e., for 0 <= x < W:
    #    - For 0 <= x < W/2: wave = 2*A_0 * (x / W)
    #    - For W/2 <= x < W: wave = 2*A_0 * (W - x) / W
    #    - For x >= W or x < 0 (shouldn't happen after mod): wave = 0

    # Create boolean masks to identify which points are in rising, falling, or outside region.
    rising_mask  = (shifted_space >= 0)   & (shifted_space < C.W / 2)
    falling_mask = (shifted_space >= C.W/2) & (shifted_space < C.W)
    # Everything else remains zero, i.e. outside [0, W].

    # Assign the rising part (0 <= x < W/2).
    molen_arr[rising_mask] = (2.0 * C.A_0 * shifted_space[rising_mask]) / C.W

    # Assign the falling part (W/2 <= x < W).
    molen_arr[falling_mask] = (2.0 * C.A_0 * (C.W - shifted_space[falling_mask])) / C.W

    return molen_arr



def Task2_caller(L, nx, t_total, dt, TimeSteppingMethod, initialisation,
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
    # initialisation      Could be:
    #  "GaussWave"          Gauassian Wave
    #  "Molenkamp"          Molenkamp triangle
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
    C.n_x = int(nx)
    C.dx = L / nx
    C.dt = dt
    C.t_total = t_total
    C.n_t = int(C.t_total / dt)
                                # first, we define a grid, or matrix, for the
                                # evolution of u in space and time;
    time = np.linspace(0, t_total, C.n_t)
    space = np.linspace(0, L, nx)
    grid = np.ndarray(shape = (nx, C.n_t), dtype = float)
    grid[:] = np.nan
    
    if initialisation == "gauss":
        grid[:, 0] = init_gauss_wave(space, L / 2, C)
    
    elif initialisation == "molen":
        grid[:, 0] = init_molenkamp_wave(space, L / 2, C)

    if DiffMethod == "CD":      # we choose the method to differentiate
        used_derivatives = derivatives
    elif DiffMethod == "PS":
        used_derivatives = derivatives_PS
    

    if TimeSteppingMethod == "Theory":
        if initialisation == "gauss":
            for idx in range(1, len(time)):
                grid[:, idx] = theoretical_gauss(space, time[idx], L / 2, C)
        elif initialisation == "molen":
            for idx in range(1, len(time)):
                grid[:, idx] = theoretical_molenkamp(space, time[idx], C)

    elif TimeSteppingMethod == "FE":
        for idx in range(1, len(time)):
            grid[:, idx] = forward_euler(grid[:, idx - 1],
                                         time[idx - 1],
                                         dt, used_derivatives, C)

    elif TimeSteppingMethod == "LF":
        for idx in range(1, len(time)):
            if idx == 1:
                grid[:, idx] = forward_euler(grid[:, idx - 1],
                                             time[idx - 1],
                                             dt, used_derivatives, C)
            else:
                grid[:, idx] = leap_frog(grid[:, idx - 2],
                                         grid[:, idx - 1],
                                         time[idx - 1],
                                         dt, used_derivatives, C)

    elif TimeSteppingMethod == "AB":
        for idx in range(1, len(time)):
            if idx == 1:
                grid[:, idx] = forward_euler(grid[:, idx - 1],
                                             time[idx - 1],
                                             dt, used_derivatives, C)
            else:
                grid[:, idx] = adam_bashforth(grid[:, idx - 2],
                                              grid[:, idx - 1],
                                              time[idx - 1],
                                              dt, used_derivatives, C)

    elif TimeSteppingMethod == "CN":
        A, B = init_crank_nicolson(C)
        for idx in range(1, len(time)):
                                # note that C is not a matrix but an instance of PhysConstants
            grid[:, idx] = crank_nicolson(grid[:, idx - 1],
                                          A, B, C)

    elif TimeSteppingMethod == "RK4":
        for idx in range(1, len(time)):
            grid[:, idx] = runge_kutta_4(grid[:, idx - 1],
                                         time[idx - 1],
                                         dt, used_derivatives, C)

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


def pprint_dict(
        d: Dict[Any, Any]
) -> None:
    """ 
    Pretty prints a dictionary

    :param d: a dictionary
    """
    for key, val in d.items():
        print(key, ' : ', val)


def main():
    # We start by calculating a lot of simulations: all algorithms for the two waves.
    # Later, we do this again but for pseudo-spectral differentiation.
    # Note that it will take approx 10 minutes to run everything (as measured on an
    # an Intel Core i7-8565U CPU, 8GB RAM, 64-bit OS)
    start_time = datetime.now()
    C = PhysConstants()
    print(f"dx = {C.dx:.6f}, dt = {C.dt:.6f}, sigma = {C.sigma:.6f}")
    
    print("\nrunning simulations for central differences...")
    print("\tGaussian wave...")
    time, space, grid_th_Gauss = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "Theory", "gauss")
    time, space, grid_FE_CD_Gauss = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "FE", "gauss", 'CD')
    time, space, grid_LF_CD_Gauss = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "LF", "gauss", 'CD')
    time, space, grid_AB_CD_Gauss = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "AB", "gauss", 'CD')
    time, space, grid_CN_CD_Gauss = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "CN", "gauss", 'CD')
    time, space, grid_RK4_CD_Gauss = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "RK4", "gauss", 'CD')
    print("\tMolenkamp wave...")
    time, space, grid_th_Molen = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "Theory", "molen")
    time, space, grid_FE_CD_Molen = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "FE", "molen", 'CD')
    time, space, grid_LF_CD_Molen = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "LF", "molen", 'CD')
    time, space, grid_AB_CD_Molen = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "AB", "molen", 'CD')
    time, space, grid_CN_CD_Molen = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "CN", "molen", 'CD')
    time, space, grid_RK4_CD_Molen = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "RK4", "molen", 'CD')

    # We store the results in a dictionary for easy access
    d_results_CD_Gauss = {
        "FE_gauss": grid_FE_CD_Gauss,
        "LF_gauss": grid_LF_CD_Gauss,
        "AB_gauss": grid_AB_CD_Gauss,
        "CN_gauss": grid_CN_CD_Gauss,
        "RK4_gauss": grid_RK4_CD_Gauss
    }
    d_results_CD_Molen = {
        "FE_molen": grid_FE_CD_Molen,
        "LF_molen": grid_LF_CD_Molen,
        "AB_molen": grid_AB_CD_Molen,
        "CN_molen": grid_CN_CD_Molen,
        "RK4_molen": grid_RK4_CD_Molen
    }
    # Some plotting preparations
    colours = ('#0A97B0', '#AE445A', '#2A3335', '#F29F58',
               '#355F2E', '#AE445A', '#A27B5C', '#FFCFEF', '#C5BAFF', '#FFEB00')
    markers = ['o', 's', '^', 'D', 'v', 'x']
    marker_ints = [2, 3, 5, 7, 11]

    print('plotting error function for Gauss and Molenkamp separately...')
    # We make an error plot for Gauss- and Molenkamp waves
    print("\tGauss...")
    d_errors_CD_Gauss = {}
    for key, val in d_results_CD_Gauss.items():
        d_errors_CD_Gauss[key] = MSE_per_t(grid_th_Gauss, val)
    for idx, (key, val) in enumerate(d_errors_CD_Gauss.items()):
        plt.plot(time, val, label = key, color = colours[idx % len(colours)],
                 marker = markers[idx % len(markers)], markevery = marker_ints[idx % len(marker_ints)])
    plt.ylim((0, MSE_per_t(grid_th_Gauss, d_results_CD_Gauss['RK4_gauss']).max()))
    plt.grid(True, which = 'both')
    plt.title('MSE(method, theory) for Gaussian wave', fontsize = 20)
    plt.xlabel('time $(s)$', fontsize = 20)
    plt.ylabel('MSE $(m^2)$', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(fontsize = 20)
    plt.show()

    # Molenkamp:
    print("\tMolenkamp...")
    d_errors_CD_Molen = {}
    for key, val in d_results_CD_Molen.items():
        d_errors_CD_Molen[key] = MSE_per_t(grid_th_Molen, val)
    for idx, (key, val) in enumerate(d_errors_CD_Molen.items()):
        plt.plot(time, val, label = key, color = colours[idx % len(colours)],
                 marker = markers[idx % len(markers)], markevery = marker_ints[idx % len(marker_ints)])
    plt.ylim((0, MSE_per_t(grid_th_Molen, d_results_CD_Molen['RK4_molen']).max()))
    plt.grid(True, which = 'both')
    plt.title('MSE(method, theory) for Molenkamp wave', fontsize = 20)
    plt.xlabel('time $(s)$', fontsize = 20)
    plt.ylabel('MSE $(m^2)$', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(fontsize = 20)
    plt.show()


    # Now, we run a stability analysis be increasing sigma logarithmically
    # and seeing how the final total error (MSE) of each simulation behaves.
    # Gaussian lines are filled and Molenkamp lines are dashed.
    print('running stability analysis...')
    waves = ['gauss', 'molen']
    methods = ['FE', 'LF', 'AB', 'CN', 'RK4']
    sigmas = np.logspace(-1, 0, 4)
    d_errors_final = {
        wave: {m: [] for m in methods} 
        for wave in waves
    }
    # We run for different sigmas and save the results
    for wave in waves:
        for method in methods:
            for sigma in sigmas:
                dt = C.dx * sigma / C.c
                print(f'\trunning for {wave}; {method}; dt = {dt}...')
                _, _, grid_th  = Task2_caller(C.L, C.n_x, C.t_total, dt, "Theory", wave)
                _, _, grid_num = Task2_caller(C.L, C.n_x, C.t_total, dt, method, wave)
                d_errors_final[wave][method].append(MSE(grid_th[:, -1], grid_num[:, -1]))
    # Now we plot them all together
    print('plotting...')
    line_styles = {'gauss': '-', 'molen': '--'}
    for idx, method in enumerate(methods):
        for wave in waves:
            plt.plot(
                sigmas,
                d_errors_final[wave][method],
                label = f'{method} - {wave}',
                color = colours[idx % len(colours)],
                marker = markers[idx % len(markers)],
                markevery = marker_ints[idx % len(marker_ints)],
                linestyle = line_styles[wave]
            )
    plt.xscale('log')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim((0, max(max(d_errors_final['gauss']['RK4']), max(d_errors_final['molen']['RK4']))))
    plt.xlabel('sigma', fontsize = 20)
    plt.ylabel('final time MSE $(m^2)$', fontsize = 20)
    plt.grid(True, which = 'both')
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(fontsize = 14)
    plt.title('sigma vs final time MSE', fontsize = 20)
    plt.show()


    # Now, pseudo-spectral differentiation. We calculate the same (but different) simulations again
    print("running simulations for pseudo-spectral differentiation...")
    print("\tGaussian wave...")
    time, space, grid_FE_PS_Gauss = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "FE", "gauss", 'PS')
    time, space, grid_LF_PS_Gauss = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "LF", "gauss", 'PS')
    time, space, grid_AB_PS_Gauss = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "AB", "gauss", 'PS')
    time, space, grid_CN_PS_Gauss = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "CN", "gauss", 'PS')
    time, space, grid_RK4_PS_Gauss = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "RK4", "gauss", 'PS')
    print("\tMolenkamp wave...")
    time, space, grid_FE_PS_Molen = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "FE", "molen", 'PS')
    time, space, grid_LF_PS_Molen = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "LF", "molen", 'PS')
    time, space, grid_AB_PS_Molen = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "AB", "molen", 'PS')
    time, space, grid_CN_PS_Molen = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "CN", "molen", 'PS')
    time, space, grid_RK4_PS_Molen = Task2_caller(C.L, C.n_x, C.t_total, C.dt, "RK4", "molen", 'PS')

    #EDIT: We did not manage to plot the pseudo-spectral derivatives in time.

    # Lastly, we make animation for both the Gaussian and Molenkamp wave
    # using normal differentiation

    ########## plotting/animating ##########

    ##### GAUSS ANIMATION ######

    print('running Gauss animation...')
    
    grids = [grid_FE_CD_Gauss, grid_LF_CD_Gauss, grid_AB_CD_Gauss, grid_CN_CD_Gauss, grid_RK4_CD_Gauss]

    fig, ax = plt.subplots()    # unpack Tuple with , for single element
    fig.subplots_adjust(left = 0.20, right = 0.95, top = 0.9, bottom = 0.15)
    line_th, = ax.plot([], [], lw = 2, label = "theoretical")
    line_FE, = ax.plot([], [], lw = 2, label = "forward Euler", marker = markers[0], markevery = marker_ints[0])
    line_LF, = ax.plot([], [], lw = 2, label = "leap-frog", marker = markers[1], markevery = marker_ints[1])
    line_AB, = ax.plot([], [], lw = 2, label = "Adams-Bashforth", marker = markers[2], markevery = marker_ints[2])
    line_CN, = ax.plot([], [], lw = 2, label = "Crank-Nicolson", marker = markers[3], markevery = marker_ints[3])
    line_RK4, = ax.plot([], [], lw = 2, label = "Runge-Kutta 4", marker = markers[4], markevery = marker_ints[4])

                                # set limits, labels, grid, legend, ticks
    ax.set_xlim(space[0], space[-1])
    ax.set_ylim(np.nanmin(grids), 1.1 * np.nanmax(grids))
    ax.set_xlabel("distance (m)", fontsize = 20)
    ax.set_ylabel("variation in u", fontsize = 20)
    ax.grid(True)
    plt.legend(fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)

    def init():                 # create empty plot animation
        line_th.set_data([], [])
        line_FE.set_data([], [])
        line_LF.set_data([], [])
        line_AB.set_data([], [])
        line_CN.set_data([], [])
        line_RK4.set_data([], [])
        return (line_th, line_FE, line_LF, line_AB, line_CN, line_RK4)

    def update(frame):          # update plot animation with new data
                                # theoretical solution
        y_th = grid_th_Gauss[:, frame]
        line_th.set_data(space, y_th)
                                # numerical solutions
        y_FE = grid_FE_CD_Gauss[:, frame]
        line_FE.set_data(space, y_FE)
        y_LF = grid_LF_CD_Gauss[:, frame]
        line_LF.set_data(space, y_LF)
        y_AB = grid_AB_CD_Gauss[:, frame]
        line_AB.set_data(space, y_AB)
        y_CN = grid_CN_CD_Gauss[:, frame]
        line_CN.set_data(space, y_CN)
        y_RK4 = grid_RK4_CD_Gauss[:, frame]
        line_RK4.set_data(space, y_RK4)

        ax.set_title(f"Gauss wave -- t = {time[frame]:.4f} s", fontsize = 20)
        return (line_FE, line_LF, line_AB, line_CN, line_RK4)

    # To vary how much frames are skipped, and thus also how fast
    # the animation moves through time, vary the \\-denominator:
    frame_skip = int(max(1, len(time) // (C.n_t / 200)))

    # https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
    ani = animation.FuncAnimation(fig,
                                    update,
                                    frames = range(0, len(time), frame_skip),
                                    init_func = init,
                                    interval = 10)
    plt.show()

    # Uncomment to save the animation
    # ani.save("gauss_advec.gif", writer = "pillow", fps = 30, dpi = 144)



    ##### MOLENKAMP ANIMATION #####
    
    print('running Molenkamp animation...')

    grids = [grid_FE_CD_Molen, grid_LF_CD_Molen, grid_AB_CD_Molen, grid_CN_CD_Molen, grid_RK4_CD_Molen]

    fig, ax = plt.subplots()    # unpack Tuple with , for single element
    fig.subplots_adjust(left = 0.20, right = 0.95, top = 0.9, bottom = 0.15)
    line_th, = ax.plot([], [], lw = 2, label = "theoretical")
    line_FE, = ax.plot([], [], lw = 2, label = "forward Euler", marker = markers[0], markevery = marker_ints[0])
    line_LF, = ax.plot([], [], lw = 2, label = "leap-frog", marker = markers[1], markevery = marker_ints[1])
    line_AB, = ax.plot([], [], lw = 2, label = "Adams-Bashforth", marker = markers[2], markevery = marker_ints[2])
    line_CN, = ax.plot([], [], lw = 2, label = "Crank-Nicolson", marker = markers[3], markevery = marker_ints[3])
    line_RK4, = ax.plot([], [], lw = 2, label = "Runge-Kutta 4", marker = markers[4], markevery = marker_ints[4])

                                # set limits, labels, grid, legend, ticks
    ax.set_xlim(space[0], space[-1])
    ax.set_ylim(np.nanmin(grids), 1.1 * np.nanmax(grids))
    ax.set_xlabel("distance (m)", fontsize = 20)
    ax.set_ylabel("variation in u", fontsize = 20)
    ax.grid(True)
    plt.legend(fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)

    def init():                 # create empty plot animation
        line_th.set_data([], [])
        line_FE.set_data([], [])
        line_LF.set_data([], [])
        line_AB.set_data([], [])
        line_CN.set_data([], [])
        line_RK4.set_data([], [])
        return (line_th, line_FE, line_LF, line_AB, line_CN, line_RK4)

    def update(frame):          # update plot animation with new data
                                # theoretical solution
        y_th = grid_th_Molen[:, frame]
        line_th.set_data(space, y_th)
                                # numerical solutions
        y_FE = grid_FE_CD_Molen[:, frame]
        line_FE.set_data(space, y_FE)
        y_LF = grid_LF_CD_Molen[:, frame]
        line_LF.set_data(space, y_LF)
        y_AB = grid_AB_CD_Molen[:, frame]
        line_AB.set_data(space, y_AB)
        y_CN = grid_CN_CD_Molen[:, frame]
        line_CN.set_data(space, y_CN)
        y_RK4 = grid_RK4_CD_Molen[:, frame]
        line_RK4.set_data(space, y_RK4)

        ax.set_title(f"Molenkamp wave -- t = {time[frame]:.4f} s", fontsize = 20)
        return (line_FE, line_LF, line_AB, line_CN, line_RK4)

    # To vary how much frames are skipped, and thus also how fast
    # the animation moves through time, vary the \\-denominator:
    frame_skip = int(max(1, len(time) // (C.n_t / 200)))

    # https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
    ani = animation.FuncAnimation(fig,
                                    update,
                                    frames = range(0, len(time), frame_skip),
                                    init_func = init,
                                    interval = 10)
    plt.show()

    # Uncomment to save the animation
    # ani.save("molenkamp_advec.gif", writer = "pillow", fps = 30, dpi = 144)
    


    print(f"total elapsed time: {datetime.now() - start_time}")


if __name__ == "__main__":
    main()