# numerieke_methoden/II/white_dwarf.py

from typing import Tuple
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
        current_state: np.array,
        current_time: float
    ) -> np.array:
    """ 
    Specific to the white dwarf problem, it returns the derivatives:
        - dx/d[ksi] = -mu/[ksi]^2 * sqrt(1 + x^2) / x; and
        - dmu/d[ksi] = x^3 * [ksi]^2.

    :param current_state: current state of the system (dependent variable)
    :param current_time: current time of the system (independent variable)
    :return: derivatives of the dependent variables
    """
    x, mu = current_state
    ksi = current_time

    if ksi == 0:            # avoid division by zero at ksi = 0
        dx_dksi = 0
    else:                   # else, just calculate the derivatives
        dx_dksi = -mu / ksi**2 * np.sqrt(1 + x**2) / x
    dmu_dksi = x**3 * ksi**2

    return np.array([dx_dksi, dmu_dksi])


def initial_conditions(
        x_center: float,
        dksi: float 
) -> Tuple[float, float]:
    """ Calculates initial conditions using a Taylor expansion
    
    :param x_center: center of the white dwarf
    :param dx: step size for the expansion
    :return: initial conditions for the white dwarf
    """                     # calculate initial x and mu, then return
    x_start = x_center - (x_center**2 / 6) * np.sqrt(1 + x_center**2) * dksi**2
    mu_start = (x_center**3 / 3) * dksi**3
    return x_start, mu_start


def integrate_white_dwarf(
        x_center: float,
        step_size: float,
        epsilon: float = 1e-12
    ) -> Tuple[np.array, np.array, np.array]:
    """ Actually integrates the white dwarf problem
    
    :param x_center: x at ksi = 0
    :param step_size: step size for the integration
    :param epsilon: threshold for the integration
    """                     # init; get initial conditions; init
    x_i = []                # x indices
    x = []                  # x values
    mu = []                 # mu values
    x_i_start, mu_start = initial_conditions(x_center, step_size)
    x_i.append(step_size)
    x.append(x_i_start)
    mu.append(mu_start)
                            # set current state and time
    current_state = np.array([x[-1], mu[-1]])
    current_time = x_i[-1]
                            # iterate until x_i > epsilon
    idx = 0
    while True:
        next_state = RK4(current_state, current_time, step_size, derivatives)

        if next_state[0] < epsilon:
            break
        
        current_time += step_size
        x_i.append(current_time)
        # print(next_state[0])
        x.append(next_state[0])
        mu.append(next_state[1])

        current_state = next_state
        
        # if idx % 1 == 0: 
        #     print(f't: {current_time}; x: {x[-1]}; mu: {mu[-1]}')
        # idx += 1

    #! TODO Maak de stopconditie wat preciezer (voor de extra opdracht)
    return np.array(x_i), np.array(x), np.array(mu)


def A() -> float:
    """ Callable for constant A
    
    :return: constant A
    """
    m_e = 9.10938356e-31    # electron mass
    c = 299792458           # speed of light
    h = 6.62607015e-34      # Planck's constant
    m_H = 1.6735575e-27     # mass of hydrogen
    return (8 * np.pi / 3) * (m_e * c / h)**3 * m_H


def P0() -> float:
    """ Callable for constant P0
    
    :return: constant P0
    """
    m_e = 9.10938356e-31    # electron mass
    c = 299792458           # speed of light
    h = 6.62607015e-34      # Planck's constant
    return (8 * np.pi / 3) * (m_e * c / h)**3 * m_e * c**2


def alpha_r() -> float:
    """ Callable for alpha_r
    
    :return: alpha_r
    """
    mu_e = 2                # mean number of nucleons per electron
    G = 6.67430e-11         # gravitational constant
    return np.sqrt(P0() / (G * mu_e**2 * A()**2 * 4 * np.pi)) 


def alpha_m() -> float:
    """ Callable for alpha_m
    
    :return: alpha_m
    """
    mu_e = 2                # mean number of nucleons per electron
    return 4 * np.pi * mu_e * A() * alpha_r()**3


def AU() -> float:
    """ Callable for atmospheric unit

    :return: one AU
    """
    return 1.9891 * 10**30


def convert_x(
        rho_center: float
    ) -> float:
    """ Converts rho [km/m^3] to its dimensionless counterpart
    
    :param rho_center: central density of the white dwarf
    :return: dimensionless rho
    """
    mu_e = 2                # mean number of nucleons per electron
    return np.cbrt(rho_center / A() * mu_e)
    

def convert_step_size(
        r: float,
) -> float:
    """ Converts step size [km] to its dimensionless counterpart
    
    :param r: step size in km
    :return: dimensionless step size
    """
    return r * 1000 / alpha_r()


def main():
    plt.figure(figsize = (8, 6))

    initial_values = np.array([[5, 300], [6, 200], [7, 150], [8, 80], [9, 20], [10, 10], [11, 5], [12, 1], [18, 1]])
    colours = ('#FFCFEF', '#0A97B0', '#AE445A', '#2A3335', '#F29F58', '#AB4459', '#441752', '#355F2E', '#FFFFFF')

    # one = 8
    # two = 80
    # x_i, x, mu = integrate_white_dwarf(convert_x(10**one), convert_step_size(two))
    # plt.plot(alpha_m() * mu / AU(),
    #          alpha_r() * x,
    #         #  color = col,
    #          label = f'x; x_c = {one}')
    # plt.xlabel('AU')
    # plt.ylabel('radius')

    for idx, col in zip(initial_values, colours):
        x_i, x, mu = integrate_white_dwarf(convert_x(10**idx[0]), convert_step_size(idx[1]))
        # plt.plot(x_i,
        #          alpha_r() * x,
        #          color = col,
        #          label = f'x; x_c = {idx[0]}')
        # plt.plot(x_i,
        #          alpha_m() * mu / AU(),
        #          color = col,
        #          label = f'mu; x_c = {idx[0]}',
        #          linestyle = '--')
        # plt.plot(alpha_m() * mu[-1] / AU(),
        #          alpha_r() * x[-1],
        #          color = col,
        #          label = f'x; x_c = {idx[0]}')
        plt.plot(alpha_m() * mu / AU(),
             alpha_r() * x,
             color = col,
             label = f'x; x_c = {idx[0]}')
        
    plt.xlabel('AU')
    plt.ylabel('radius')
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == '__main__':
    main()