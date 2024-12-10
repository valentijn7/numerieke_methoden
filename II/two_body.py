# numerieke_methoden/II/two_body.py

import numpy as np
import matplotlib.pyplot as plt


def RK4(
        current_state: np.array,
        current_time: float,
        step_size: float,
        derivative: callable
    ) -> np.array:
    """ Performs one step of the 4th order Runge-Kutta
    
    :param current_state: current state of the system (dependent variable)
    :param current_time: current time of the system (independent variable)
    :param step_size: step size of the integration
    :param derivative: function that returns derivatives for current state and time
    :return: updated values of dependent variable at next time step
    """
                            # calculate the 4 k's
    k1 = step_size * derivative(current_state, current_time)
    k2 = step_size * derivative(current_state + 0.5 * k1, current_time + 0.5 * step_size)
    k3 = step_size * derivative(current_state + 0.5 * k2, current_time + 0.5 * step_size)
    k4 = step_size * derivative(current_state + k3, current_time + step_size)

                            # calculate and return new state
    return current_state + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def derivatives(
        state: np.array,
        time: float,
        c_s, Dict[str, float]
    ) -> np.array:
    """ Specific to the two-body problem, using the equations:
        - (26) d vec(r)i/dt = vi; and
        - (27) d vec(v)i/dt = sum(j = 1; j != i)(GMj * (vec(r)j - vec(ri)) / 
                                                ((xi - xj)^2 + (yi - yj)^2)^(3/2),
        with vec(r)i = (xi, yi) and vec(v)i = (vxi, vyi).

    :param state: current state of the system (dependent variable)
    :param time: current time of the system (independent variable)
    :param c_s: dictionary with constants
    :return: derivatives of the dependent variables
    """
    G, M1, M2 = c_s['G'], c_s['M1'], c_s['M2']
    rx, ry, vx, vy = state
    distance_cubed = (rx**2 + ry**2)**(3/2)

    factor = -(G * M2 / (1 + M1 / M2))
    dvx_dt = factor * (rx / distance_cubed)
    dvy_dt = factor * (ry / distance_cubed)

    return np.array([vx, vy, dvx_dt, dvy_dt])


def main():
    constants = {
        'G': 6.67430e-11,  
        'M1': 5.972e24,  
        'M2': 1.989e30,
    }



if __name__ == '__main__':
    main()