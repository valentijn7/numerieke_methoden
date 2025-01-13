# numerieke_methoden/III/Task1_MandatoryStructure.py

from typing import List, Dict, Tuple, Any, Callable
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from datetime import datetime

start_time = datetime.now()

class PhysConstants:
    def __init__(self):
        self.kappa   = 5        # thermal diffusion coefficient (m2/s)
                                #   temperatures:
        self.T0      = 273      # initial temperature rod (K)
        self.T1      = 373      # temperature of rod at x = 0 for t > 0
                                #   rod properties:
        self.L       = 10       # length of the rod (m)
        self.n_x     = 101      # number of gridpoints in the rod
                                # step size of the grid
        self.dx      = self.L / self.n_x
                                #   simulation properties:
        self.t_total = 0.3      # total time of the simulation (s)
        self.n_t     = 3001     # number of timesteps
                                # time step of the simulation (s)
        self.dt      = self.t_total / (self.n_t - 1)


def forward_euler(
        current_state: np.array,
        current_time: float,
        step_size: float,
        derivative: callable,
        C: PhysConstants
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
        C: PhysConstants
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
        C: PhysConstants
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
        cons: PhysConstants
    ) -> np.array:
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


def calculate_sigma_ratio(
        C: PhysConstants
    ) -> float:
    """ Calculates the ratio of the time step to the spatial step

    :param C: class (struct) with constants
    :return: ratio of the time step to the spatial step
    """
    return C.kappa * C.dt / C.dx**2


def Task1_caller(
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
    nt = int(t_total / dt + 1)
    dx = L / (nx - 1)
    global sigma
    sigma = (C.kappa * dt) / dx**2
    
                                # first, we define a grid, or matrix, for space and time;
    grid = np.ndarray(shape = (nx, nt), dtype = float)
    grid[:] = np.nan
    grid[:, 0] = C.T0
    grid[0, :] = C.T1
    grid[:, -1] = 0.0 

                                # second, we define the space and time arrays
    # global time and space
    time = np.arange(0, t_total + t_total / (nt - 1), t_total / (nt - 1))
    space = np.arange(0, L + dx, dx)
    
    global timespacing
    timespacing = int(nt / 5)                            
    # third, solve heat equation
    if TimeSteppingMethod == "Theory":
        for idx in range(1, len(time)):
            grid[:, idx] = theoretical(space, time[idx], C)
        global T_arr_th
        T_arr_th = grid[:,::timespacing].copy() # For plotting the first couple of time-values

    elif TimeSteppingMethod == "FE":
        for idx in range(1, len(time)):
            grid[:, idx] = forward_euler(grid[:, idx - 1], time[idx - 1], dt, derivative_heat, C)
        global T_arr_FE
        T_arr_FE = grid[:,::timespacing].copy() 

    elif TimeSteppingMethod == "LF":
        for idx in range(1, len(time)):
            if idx == 1:
                grid[:, idx] = forward_euler(grid[:, idx - 1], time[idx - 1], dt, derivative_heat, C)
            else:
                grid[:, idx] = leap_frog(grid[:, idx - 2], grid[:, idx - 1], time[idx - 1], dt, derivative_heat, C)
        global T_arr_LF
        T_arr_LF = grid[:,::timespacing].copy()
        
    elif TimeSteppingMethod == "AB":
        for idx in range(1, len(time)):
            if idx == 1:
                grid[:, idx] = forward_euler(grid[:, idx - 1], time[idx - 1], dt, derivative_heat, C)
            else:
                grid[:, idx] = adam_bashforth(grid[:, idx - 2], grid[:, idx - 1], time[idx - 1], dt, derivative_heat, C)
        global T_arr_AB
        T_arr_AB = grid[:,::timespacing].copy()
        
    elif TimeSteppingMethod == "RK4":
        for idx in range(1, len(time)):
            grid[:, idx] = RK4(grid[:, idx - 1], time[idx - 1], dt, derivative_heat, C)
        global T_arr_RK4
        T_arr_RK4 = grid[:,::timespacing].copy()
        
    # elif TimeSteppingMethod == "CN":
    #     for idx in range(1,len(time)):
            
    else:
        raise ValueError("Invalid TimeSteppingMethod")
    
    # Now we plot our data. We will plot the theoretical data vs. the 
    # numerical data seperately, i.e. we get 5 plots.
    
    
    
    return time, space, grid   


#%%
def main():
    #! TODO bereken die ratio van de lecture slides
    C = PhysConstants()
    sigma = calculate_sigma_ratio(C)
    print(f"sigma = {sigma:.4f}")



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
    
    time, space, grid_RK4 = Task1_caller(
        C.L,
        C.n_x,
        C.t_total,
        C.dt,
        "RK4")
    
    ######### Plotting #########

    grids = [grid_th, grid_FE, grid_LF, grid_AB, grid_RK4]

    fig, ax = plt.subplots()    # unpack Tuple with , for single element
    line_th, = ax.plot([], [], lw = 2, label = "theoretical")
    line_FE, = ax.plot([], [], lw = 2, label = "forward Euler")
    line_LF, = ax.plot([], [], lw = 2, label = "leap-frog")
    line_AB, = ax.plot([], [], lw = 2, label = "Adams-Bashforth")
    line_RK4, = ax.plot([], [], lw = 2, label = "Runge-Kutta 4")

                                # set limits, labels, grid, legend, ticks
    ax.set_xlim(space[0], space[-1])
    ax.set_ylim(np.nanmin(grids), np.nanmax(grids))
    ax.set_xlabel("rod (m)", fontsize = 18)
    ax.set_ylabel("temperature (K)", fontsize = 18)
    plt.legend(fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    ax.grid(True)

    def init():                 # create empty plot animation
        line_FE.set_data([], [])
        line_LF.set_data([], [])
        line_AB.set_data([], [])
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
        y_RK4 = grid_RK4[:, frame]
        line_RK4.set_data(space, y_RK4)

        ax.set_title(f"t = {time[frame]:.4f} s", fontsize = 18)
        return line_FE, line_LF, line_AB, line_RK4, line_th

    # https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
    ani = animation.FuncAnimation(fig,
                                  update,
                                  frames = len(time),
                                  init_func = init,
                                  interval = 100)

    plt.show()
    
    # now, we plot our data in stationary graphs. We will plot theoretical vs
    # numerical for each method, resulting in 5 plots different plots
    
    global methods_list
    methods_list = [T_arr_FE, T_arr_LF, T_arr_AB, T_arr_RK4] # and CN once we get it
    
    name_list = ['Forward Euler', 'Leap Frog', 'Adams-Bashforth', 'Runge-Kutta-4', 'Crank-Nicholson']
    
    name_counter = 0
    for T_results in methods_list:
        plt.figure()
        for index in range(1,T_results.shape[1]):
            plt.plot(space, T_results[:, index], label = f't = {timespacing * C.dt * index}')
            plt.plot(space, T_arr_th[:, index], linestyle = ':')
            #TODO: sigma onderzoeken, plotjes werken nu (behalve leap-frog, wss ivm sigma)
        
        plt.xlabel('position on rod $(m)$', fontsize = 18)
        plt.ylabel('temperature $(K)$', fontsize = 18)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.legend()
        plt.title(f'{name_list[name_counter]} vs. theory')
        name_counter += 1
        plt.show()
        

if __name__ == "__main__":
    main()


print(f"\ntotal runtime: {datetime.now() - start_time}")