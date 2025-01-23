# numerieke_methoden/III/Navier-Stokes/task3.py

from typing import List, Dict, Tuple, Any, Callable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from algorithms import RK4


# Note that every time the term "velocity" is used in this script,
# what is meant is horizontal velocity.


class PhysConstants:
    def __init__(self):         
                                #   constants:
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
    

def Task3_caller(
        L: int,
        nx: int,
        t_total: int,
        dt: int,
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
                    
                                # init a normally distributed grid
    height = 5
    h_grid[:, 0] = init_Gaussian_wave((C.n_x + 1) / 2, 20, C) + height

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


def main():
    C = PhysConstants()
    C.L = 10
    C.n_x = 200
    C.t_total = 1
    C.dt = 1e-5

    time, space, h_grid, u_grid = Task3_caller(
        C.L,
        C.n_x,
        C.t_total,
        C.dt
    )

    colours = ('#0A97B0', '#AE445A', '#2A3335', '#F29F58',
               '#355F2E', '#AE445A', '#A27B5C', '#FFCFEF')
    
    plt.figure(figsize = (10, 5))

    for t, col in zip(range(0, C.n_t, int(C.n_t / 5)), colours):
        plt.plot(space, h_grid[:, t], marker = '.', color = col,
                 label = f't = {time[t]:.4f}s')

    plt.title(f'water distribution over time for RK4', fontsize = 20)
    plt.xlabel('position x $(m)$', fontsize = 20)
    plt.ylabel('height H $(m)$', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(fontsize = 20)
    plt.grid()
    plt.show()


    # Wave animation:

    fig, ax = plt.subplots()    
                                # unpack Tuple with ',' for single element
    line, = ax.plot([], [], lw = 2, label = "RK4", color = '#0A97B0')

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
        ax.set_title(f"t = {time[frame]:.4f} s", fontsize = 20)
        return (line, )

    frame_skip = int(max(1, len(time) // (C.n_t / 5)))
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
    ani = animation.FuncAnimation(fig,
                                  update_wave,
                                  frames = range(0, len(time), frame_skip),
                                  init_func = init_wave,
                                  interval = 1,
                                #   blit = True,
                                  repeat = True
                                  )
    plt.show()


if __name__ == '__main__':
    main()