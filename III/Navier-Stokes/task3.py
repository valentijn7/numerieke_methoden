# numerieke_methoden/III/Navier-Stokes/task3.py

from typing import List, Dict, Tuple, Any, Callable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime

from algorithms import RK4

# By default, we've commented out the experiments as they
# have a long runtime. To run them, simply uncomment the
# function calls at the bottom of the script.

# Note that every time the term "velocity" is used in this script,
# what is meant is horizontal velocity.


class PhysConstants:
    def __init__(self):         
                                #   constants:
        self.g       = 9.81     # gravitational acceleration
        self.b       = 1        # height of water at rest (constant)
        self.beta    = 0.01     # the constant parametrising friction

                                #   water column properties:
        self.L       = 10       # horizontal length of the column (m)
        self.n_x     = 200      # number of spatial steps
                                # step size of the grid
        self.dx      = self.L / self.n_x

                                #   simulation properties:
        self.t_total = 2        # total time of the simulation (s)
        self.n_t     = 100000   # number of time steps
                                # time step of the simulation (s)
        self.dt      = self.t_total / self.n_t

# We initially considered to use a global instance of PhysConstants
# for all simulations, but do the many different natures of them
# (e.g. caused by the experiments) we now use multiple C instantiations
# throughout the script.


def CD(
        state: np.ndarray, 
        step_size: float
    ) -> np.array:
    """
    Uses central differentiation to differentiate a given input array;
    at the ends, forward and backward differentiation are used.
    
    :param state: input array of the variable
    :param step_size: size of the step over which is differentiated
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
    dH_dt = -CD(H_u, dx)
    return dH_dt


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
    return -CD(helper_var, dx) - C.beta * (velocity / height)


def init_Gaussian_wave(
        midpoint: float,
        std: float,
        C: Any
    ) -> np.array:
    """ Inits an array with a Gaussian curvature

    :param midpoint: midpoint
    :param std: std
    :param C: class (struct) with hyperparams
    :return: array with Gaussian curvature
    """ 
    array = np.ndarray((C.n_x + 1), dtype = float)
    for idx, _ in enumerate(array):
        array[idx] = np.e**(-(0.5) * (idx - midpoint)**2 / (std)) \
                     / (std * np.sqrt(2 * np.pi))
    return array


def init_steep_wave(
        rest_height: float,
        wave_height: float,
        steps_to_height: int,
        nx: int
    ) -> np.ndarray:
    """ Creates a "steep" wave; that is, an array of heights which starts
    at a rest height, climbs steeply to a wave height, and then continues
    at that height until the end of the array. The steepness is controlled
    by the number of steps it takes to reach the wave height
    
    :param rest_height: height of water on the left
    :param wave_height: height of the wave
    :param transition_steps: transition steps
    :return: array with steep wave
    """
    steps_to_height += 1        # to negate effects of exclusion
                                # of ranges in fcts
    if steps_to_height > nx + 1:
        raise ValueError("\ntransition cannot be larger than length\n")
    
                                # init the wave into 'arr'
    arr = np.zeros((nx + 1), dtype = float)
    arr[: int(nx / 2)] = rest_height
    arr[int(nx / 2) :] = wave_height
    
    height_diff = wave_height - rest_height
    if height_diff == 0:
        raise ValueError("\nheight difference cannot be zero\n")
    elif height_diff < 0:
        raise ValueError("\nrest height cannot be higher than wave height\n")
    
    slope = height_diff / (steps_to_height - 1)
    print(f"init_steep_wave(): slope = {slope}")

    arr[int((nx + 1 - steps_to_height) / 2): \
        int((nx + 1 + steps_to_height) / 2)] = \
            np.linspace(
                rest_height, 
                wave_height, 
                steps_to_height
            )

    return arr


def Task3_caller(
        L: int,
        nx: int,
        t_total: int,
        dt: int,
        height: float,
        max_height: float,
        initial_wave_type: str
    ) -> Tuple[np.array, np.array, np.array]:
    """ This routine calls the function that solves the heat equation

    The mandatory input is:
    L                   Length of domain to be modelled (m)
    nx                  Number of gridpoint in the model domain
    t_total             Total length of the simulation (s)
    dt                  Length of each time step (s)
    initial_wave_type:
        "gauss" = Creates Gaussian wave as initial input
        "steep" = Creates steep wave as initial input
    
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
                    
                                # init a normally distributed grid
    if initial_wave_type == "gauss":
        h_grid[:, 0] = init_Gaussian_wave((C.n_x + 1) / 2, 20, C) + height
    elif initial_wave_type == "steep":
        steps_to_height = int((C.n_x+1)/5)
        h_grid[:, 0] = init_steep_wave(height, max_height, steps_to_height, nx)
                                # init u grid with 1, and set u to 0 on boundaries
    u_grid = np.zeros((nx + 1, C.n_t + 1))
    u_grid[0, :] = 0.0
    u_grid[-1, :] = 0.0
                                # fill Hu grid with product of h and u
    Hu_grid = u_grid * h_grid
                                # b array is filled with constant value
    b = np.full(nx + 1, C.b)
                                # create time and space arrays
    time = np.linspace(0, t_total, C.n_t + 1)
    space = np.linspace(0, L, nx + 1)

                                # Iterate over time!
    for idx in range(1, len(time)):
                                # we fix the velocity at the boundaries
        u = Hu_grid[:, idx - 1] / h_grid[:, idx - 1]
        u[0] = 0.0
        u[-1] = 0.0

        h_grid[:, idx] = RK4(
            h_grid[:, idx - 1],
            h_grid[:, idx - 1],
            u, b, C.dt, C.dx, H_time_deriv, C
        )
        Hu_grid[:, idx] = RK4(
            Hu_grid[:, idx - 1],
            h_grid[:, idx - 1],
            u, b, C.dt, C.dx, Hu_time_deriv, C
        )

    return time, space, h_grid, Hu_grid / h_grid


def dt_tester(
        dt_min: float, 
        dt_max: float, 
        steps: int, 
        step_manner: str, 
        t_total: float
    ) -> None:
    """ With this comprehensive function, we do a test on the stability
    of the simulation by trying various dt-values between dt_min and dt_max.
    We measure the volume at the start and end of the simulation, and check
    the difference. (The assumption is that unstable methods will not have
    conservation of mass (Lavoisier).) This method is not 100% robust, but
    it should be a pretty good indicator of stability.

    :param dt_min: minimal dt over which testing is done
    :param dt_max: maximal dt over which testing is done
    :param steps: how many dt's are tested between dt_min and dt_max
    :param step_manner: string of either "lin" or "log", dictating the way
                        different dt-values are chosen between dt_min and dt_max
    :param t_total: total simulation time
    :return: None --> results are printed/plotted
    """
    start_time_dt = datetime.now()
    print("running the dt-tester...\n")

    C = PhysConstants()
    C.L = 10    
    C.n_x = 200
    C.t_total = t_total
    C.n_t = int(C.t_total / C.dt)
    C.dx = C.dx
    
    if step_manner == "lin":    # create array of dt-values to test
                                # with either linear or log spacing
        dt_guesses = np.linspace(
            dt_min, dt_max, steps
        )
    elif step_manner == "log":
        dt_guesses = np.logspace(
            np.log10(dt_min), np.log10(dt_max), steps
        )
        
    if np.any(dt_guesses > t_total):
        raise ValueError("stability analysis: "
                         "dt cannot be bigger than total simulation time")
    
    volume_anomalies = []
    added_labels = []
    
    for dt in dt_guesses:
        print(f"\n\trunning with dt = {dt}...\n")
        
        time, space, h_grid, u_grid = Task3_caller(
            C.L,
            C.n_x,
            C.t_total,
            dt,
            5,
            10,
            "gauss")
                                # check for NaNs and infs in the grid
        if np.isnan(h_grid).any() or np.isinf(abs(h_grid)).any():
            print(f"found NaN/inf in h_grid for dt = {dt}\n")
            volume_anomalies.append(1)

            if 'NaN/inf' not in added_labels:
                added_labels.append('NaN/inf')
                plt.plot(dt, 1,
                        marker = 'x', markersize = 15, color = 'r',
                        label = 'NaN/inf')
            else:
                plt.plot(dt, 1,
                        marker = 'x', markersize = 15, color = 'r')

            continue            # skip to next iteration

                                # we calculate the volume of water at the
                                # start and end of the simulation, and check
                                # the difference
        water_volume_start = (h_grid[:, 0] * C.dx).sum()
        water_volume_end = (h_grid[:, -1] * C.dx).sum()
        difference = water_volume_end - water_volume_start

                        
        if difference == 0:     # three cases: no anomaly;
                                # positive anomaly; negative anomaly
            volume_anomalies.append(difference)
            
            print(f"no volume anomaly found for dt = {dt}")
            if 'no anomaly' not in added_labels:
                added_labels.append('no anomaly')
                plt.plot(dt, 1,
                         marker = '*', markersize = 15, color = 'green',
                         label = 'no anomaly')
            else:
                plt.plot(dt, 1,
                            marker = '*', markersize = 15, color = 'green')
    
        elif difference < 0:
            volume_anomalies.append(difference)

            print(f"water volume anomaly for dt = {dt} is negative \n")
            if 'negative anomaly' not in added_labels:
                added_labels.append('negative anomaly')
                plt.plot(dt, abs(difference),
                         marker = 'v', markersize = 15, color = 'blueviolet',
                         label = 'negative anomaly')
            else:
                plt.plot(dt, abs(difference),
                        marker = 'v', markersize = 15, color = 'blueviolet')
        else:
            volume_anomalies.append(difference)

            print(f"water volume anomaly for dt = {dt} is positive \n")
            if 'positive anomaly' not in added_labels:
                added_labels.append('positive anomaly')
                plt.plot(dt, difference,
                         marker = '.', markersize = 15, color = 'royalblue',
                         label = 'positive anomaly')
            else:
                plt.plot(dt, difference,
                         marker = '.', markersize = 15, color = 'royalblue')

   
    plt.title(f'water volume anomaly vs. dt; simulation time = 1s', fontsize = 20)
    plt.xlabel('chosen time intervals dt $(s)$', fontsize = 20)
    plt.ylabel('volume anomaly $(m^2)$', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which = 'both')
    # plt.ylim(0.5 * np.min(volumes), 3 * np.max(volumes))
    plt.savefig('volume_anomaly.pdf', dpi = 300, bbox_inches = 'tight',
                pad_inches = 0.5)
    plt.legend(fontsize = 20, facecolor = '#F0F0F0')
    plt.show()

    print(f"\ntotal elapsed time for dt-tester = {datetime.now() - start_time_dt}")


def exp_steep_wave() -> animation.FuncAnimation:
    """ This function runs the simulation for a steep wave, and
    plots the results. The wave starts at a rest height of 5,
    and climbs steeply to a height of 10. The simulation is
    run for 100 seconds, with a time step of 1e-4 seconds.
    """
    start_time_esw = datetime.now()
    plt.close('all')
    
    C = PhysConstants()
    C.L = 10
    C.n_x = 200
    C.t_total = 1
    C.dt = 1e-5
    C.n_t = int(C.t_total / C.dt)
    
    # rest_height = 5
    # test_heights = [6, 7, 8, 9]
    rest_height = 1
    test_heights = [1.5, 2, 2.5, 3]
    # We will run the experiment for 4 different wave heights;
    # it should take about a minute to generate the animation
    # (while outputting a lot of runtime warnings)
    time, space, h_grid_1, u_grid_1 = Task3_caller(
        C.L,
        C.n_x,
        C.t_total,
        C.dt,
        rest_height,
        test_heights[0],
        "steep"
    )
    time, space, h_grid_2, u_grid_2 = Task3_caller(
        C.L,
        C.n_x,
        C.t_total,
        C.dt,
        rest_height,
        test_heights[1],
        "steep"
    )
    time, space, h_grid_3, u_grid_3 = Task3_caller(
        C.L,
        C.n_x,
        C.t_total,
        C.dt,
        rest_height,
        test_heights[2],
        "steep"
    )
    time, space, h_grid_4, u_grid_4 = Task3_caller(
        C.L,
        C.n_x,
        C.t_total,
        C.dt,
        rest_height,
        test_heights[3],
        "steep"
    )
    h_grids = [h_grid_1, h_grid_2, h_grid_3, h_grid_4]
    
    # Wave animation:

    fig, ax = plt.subplots()    
                                # set limits, labels, grid, legend, ticks
    ax.set_xlabel("position x $(m)$", fontsize = 20)
    ax.set_ylabel("height H $(m)$", fontsize = 20)
    ax.set_xlim(space.min(), space.max())
    # lines below are to ensure the min- and max-values are plottable numbers
    MAX_VALUE = 20
    MIN_VALUE = 0
    for idx, grid in enumerate(h_grids):
        grid = np.where(np.isinf(grid), np.nan, grid)
        grid = np.where(np.abs(grid) > MAX_VALUE, np.nan, grid)
        grid = np.where(np.abs(grid) < MIN_VALUE, np.nan, grid)
        h_grids[idx] = grid
    ax.set_ylim(np.nanmin(h_grids), np.nanmax(h_grids))
    ax.grid(True, which = 'both')
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)

    # unpack Tuple with ',' for single element
    line_1, = ax.plot([], [], lw = 2,
                      label = f"top wave height = {test_heights[0]}", color = '#D5E7B5')
    line_2, = ax.plot([], [], lw = 2,
                      label = f"top wave height = {test_heights[1]}", color = '#72BAA9')
    line_3, = ax.plot([], [], lw = 2,
                      label = f"top wave height = {test_heights[2]}", color = '#7E5CAD')
    line_4, = ax.plot([], [], lw = 2,
                      label = f"top wave height = {test_heights[3]}", color = '#474E93')
    plt.legend(fontsize = 20, facecolor = '#F0F0F0')

    def init_wave():
        """" Create empty plot animation """
        line_1.set_data([], [])
        line_2.set_data([], [])
        line_3.set_data([], [])
        line_4.set_data([], [])
        return (line_1, line_2, line_3, line_4 )

    def update_wave(frame):
        """ Update plot animation with new data """
        line_1.set_data(space, h_grid_1[:, frame])
        line_2.set_data(space, h_grid_2[:, frame])
        line_3.set_data(space, h_grid_3[:, frame])
        line_4.set_data(space, h_grid_4[:, frame])
        ax.set_title(f"1D SWE for rest height of {rest_height} -- "
                     f"t = {time[frame]:.4f} s", fontsize = 20)
        return (line_1, line_2, line_3, line_4 )

    frame_skip = int(max(1, len(time) // (C.n_t / 1000)))
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
    anim = animation.FuncAnimation(fig,
                                  update_wave,
                                  frames = range(0, len(time), frame_skip),
                                  init_func = init_wave,
                                  interval = 1,
                                #   blit = True,
                                  repeat = True
                                  )
    print(f"total runtime steep wave experiment: {datetime.now() - start_time_esw}")
    # fig.subplots_adjust(left = 0.25, right = 0.95, top = 0.9, bottom = 0.15)
    # anim.save('steep_wave_animation.gif', writer = 'pillow', fps = 30, dpi = 144)
    plt.show()

    # returning the animation object got rid of errors on one of our machines,
    # although we're not sure why. We keep it in for now to be safe!
    return anim


def exp_low_heights() -> animation.FuncAnimation:
    """ This function runs an experiment where we have a steep wave
    starting on different starting heights. We want to see how low
    the starting height can be before the simulation becomes unstable.
    """
    start_time_lh = datetime.now()
    plt.close('all')

    C = PhysConstants()        # init PhysConstants and recalculate based on input params   
    C.L = 10
    C.n_x = 200
    C.t_total = 20
    C.dt = 1e-4
    C.n_t = int(C.t_total / C.dt)
    
    max_height = 0.2
    test_low_heights = [1e-1, 1e-2, 1e-3]
    # We will run the experiment for 4 different heights;
    # it takes around 3 minutes to generate the animation
    time, space, h_grid_1, u_grid_1 = Task3_caller(
        C.L,
        C.n_x,
        C.t_total,
        C.dt,
        test_low_heights[0],
        max_height,
        "steep"
    )
    
    time, space, h_grid_2, u_grid_2 = Task3_caller(
        C.L,
        C.n_x,
        C.t_total,
        C.dt,
        test_low_heights[1],
        max_height,
        "steep"
    )
    
    time, space, h_grid_3, u_grid_3 = Task3_caller(
        C.L,
        C.n_x,
        C.t_total,
        C.dt,
        test_low_heights[2],
        max_height,
        "steep"
    )
    
    # time, space, h_grid_4, u_grid_4 = Task3_caller(
    #     C.L,
    #     C.n_x,
    #     C.t_total,
    #     C.dt,
    #     test_low_heights[3],
    #     max_height,
    #     "steep"
    # )
    
    h_grids = [h_grid_1, h_grid_2, h_grid_3]
    
    # Wave animation:

    fig, ax = plt.subplots()    
                                # set limits, labels, grid, legend, ticks
    ax.set_xlabel("position x $(m)$", fontsize = 20)
    ax.set_ylabel("height H $(m)$", fontsize = 20)
    ax.set_xlim(space.min(), space.max())
    # lines below are to ensure the min- and max-values are plottable numbers
    MAX_VALUE = 20
    MIN_VALUE = 0
    for idx, grid in enumerate(h_grids):
        grid = np.where(np.isinf(grid), np.nan, grid)
        grid = np.where(np.abs(grid) > MAX_VALUE, np.nan, grid)
        grid = np.where(np.abs(grid) < MIN_VALUE, np.nan, grid)
        h_grids[idx] = grid
    ax.set_ylim(np.nanmin(h_grids), np.nanmax(h_grids))
    ax.grid(True, which = 'both')
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.yscale('log')

    # unpack Tuple with ',' for single element
    line_1, = ax.plot([], [], lw = 2,
                      label = f"rest height = {test_low_heights[0]}", color = '#0A97B0')
    line_2, = ax.plot([], [], lw = 2,
                      label = f"rest height = {test_low_heights[1]}", color = '#AE445A')
    line_3, = ax.plot([], [], lw = 2,
                      label = f"rest height = {test_low_heights[2]}", color = '#2A3335')
    # line_4, = ax.plot([], [], lw = 2,
    #                   label = f"rest height = {test_low_heights[3]}", color = '#F29F58')
    plt.legend(fontsize = 20, facecolor = '#F0F0F0')

    def init_wave():
        """" Create empty plot animation """
        line_1.set_data([], [])
        line_2.set_data([], [])
        line_3.set_data([], [])
        # line_4.set_data([], [])
        # return (line_1, line_2, line_3, line_4)
        return (line_1, line_2, line_3, )

    def update_wave(frame):
        """ Update plot animation with new data """
        line_1.set_data(space, h_grid_1[:, frame])
        line_2.set_data(space, h_grid_2[:, frame])
        line_3.set_data(space, h_grid_3[:, frame])
        # line_4.set_data(space, h_grid_4[:, frame])
        ax.set_title(f"1D SWE for rest height of 5 -- "
                     f"t = {time[frame]:.4f} s", fontsize = 20)
        # return (line_1, line_2, line_3, line_4 )
        return (line_1, line_2, line_3, )

    frame_skip = int(max(1, len(time) // (C.n_t / 1000)))
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
    anim = animation.FuncAnimation(fig,
                                  update_wave,
                                  frames = range(0, len(time), frame_skip),
                                  init_func = init_wave,
                                  interval = 1,
                                #   blit = True,
                                  repeat = True
                                  )
    print(f"total runtime low heights experiment: {datetime.now() - start_time_lh}")
    # fig.subplots_adjust(left = 0.25, right = 0.95, top = 0.9, bottom = 0.15)
    # anim.save('low_heights_animation.gif', writer = 'pillow', fps = 30, dpi = 144)
    plt.show()

    # returning the animation object got rid of errors on one of our machines,
    # although we're not sure why. We keep it in for now to be safe!
    return anim


def main():             # script starts here!
                        # generating all plots takes 10 minutes approx.;
                        # leaving out the experiments, this is reduced to 4 minutes.
    start_time = datetime.now()
    print("running the main script...\n")	

    C = PhysConstants() # init a PhysConstants object with constants
    
    print("running 1st simulation...\n")
    time, space, h_grid, u_grid = Task3_caller(
        C.L,
        C.n_x,
        C.t_total,
        C.dt,
        5,
        10,
        "gauss"
    )

    colours = ('#0A97B0', '#AE445A', '#2A3335', '#F29F58',
               '#355F2E', '#663399', '#A27B5C', '#FFCFEF')
    
    print("creating 1st figure...\n")
    plt.figure(figsize = (10, 5))
    for t, col in zip(range(0, C.n_t, int(C.n_t / 8)), colours):
        plt.plot(space, h_grid[:, t], 
                 marker = '.',
                 color = col,
                 label = f't = {time[t]:.4f}s')

    plt.title(f'water distribution over time for RK4', fontsize = 20)
    plt.xlabel('position x $(m)$', fontsize = 20)
    plt.ylabel('height H $(m)$', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(fontsize = 20, loc = 'upper right')
    plt.grid()
    plt.show()


    ##### wave animation #####

    # this takes approx. 2:30 minutes! 
    # (processed on an Intel Core i7-8565U CPU, 8GB RAM, 64-bit OS)
    print("creating standard wave animation...\n")

    C.t_total = 10              # take more time for the simulation
    C.n_t = 500000              # to see a bit of "chaos" in the end
    time, space, h_grid, u_grid = Task3_caller(
        C.L,
        C.n_x,
        C.t_total,
        C.dt,
        5,
        10,
        "gauss"
    )

    fig, ax = plt.subplots()    
                                # unpack Tuple with ',' for single element
    line, = ax.plot([], [], lw = 2, label = "RK4", color = 'mediumblue')

                                # set limits, labels, grid, legend, ticks
    ax.set_title('1D Navier-Stokes wave simulation', fontsize = 20)
    ax.set_xlabel("position x $(m)$", fontsize = 20)
    ax.set_ylabel("height H $(m)$", fontsize = 20)
    ax.set_xlim(space.min(), space.max())
    ax.set_ylim(np.nanmin(h_grid), np.nanmax(h_grid))
    ax.grid(True, which = 'both')
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(fontsize = 20, facecolor = '#F0F0F0')


    def init_wave():
        """" Create empty plot animation """
        line.set_data([], [])
        return (line, )

    def update_wave(frame):
        """ Update plot animation with new data """
        line.set_data(space, h_grid[:, frame])
        ax.set_title(f"t = {time[frame]:.4f}s", fontsize = 20)
        return (line, )
    
                                # to vary the sampling rate, and thus how
                                # fast the simulation "moves" through time,
                                # vary the denominator of the frame_skip
    frame_skip = int(max(1, len(time) // (C.n_t / 2500)))

    # https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
    anim = animation.FuncAnimation(fig,
                                  update_wave,
                                  frames = range(0, len(time), frame_skip),
                                  init_func = init_wave,
                                  interval = 1,
                                #   blit = True,
                                  repeat = True
                                  )
    # fig.subplots_adjust(left = 0.25, right = 0.95, top = 0.9, bottom = 0.15)
    # anim.save('gaussian_wave_animation.gif', writer = 'pillow', fps = 30, dpi = 144)
    plt.show()



    ##### stability analysis #####
    # we test the stability by doing a simulation with various dt-values

    print("running the stability analysis...\n")	
    dt_tester(
        dt_min = 1e-5,  # minimal dt value to try (any values lower than this were found stable)
        dt_max = 1e-3,  # maximal dt value to try
        steps = 7,      # how many steps to take between dt_min and dt_max;
                        # (we take 7 for low computational cost, but the pattern
                        # should be clear)
                        # how to take the steps between dt_min and dt_max
        step_manner = "log",
                        # total simulation time in seconds
        t_total = 1 
    )

    ##### steep wave experiment #####
    # our first experiment is to see how the simulation behaves with a very steep wave;
    # uncomment/comment it to run/hide it

    print("running the steep wave experiment...\n")
    anim = exp_steep_wave()


    ##### low heights experiment #####
    # our second experiment is to see how the simulation behaves with very low wave heights;
    # uncomment/comment it to run/hide it

    print("running the low heights experiment...\n")
    anim = exp_low_heights()


    print(f'total time elapsed: {datetime.now() - start_time}')
    

if __name__ == '__main__':
    main()