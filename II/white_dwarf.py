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
        current_ksi: float
    ) -> np.array:
    """ 
    Specific to the white dwarf problem, it returns the derivatives:
        - dx/d[ksi] = -mu/[ksi]^2 * sqrt(1 + x^2) / x; and
        - dmu/d[ksi] = x^3 * [ksi]^2.

    :param current_state: current state of the system (dependent variable)
    :param current_ksi: current time of the system (independent variable)
    :return: derivatives of the dependent variables
    """
    x, mu = current_state
    ksi = current_ksi

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
        step_size: float,
        x_center: float,
        epsilon: float = 1e-12,
        verbose: bool = False
    ) -> Tuple[np.array, np.array, np.array]:
    """ Actually integrates the white dwarf problem
    
    :param x_center: x at ksi = 0
    :param step_size: step size for the integration
    :param epsilon: threshold for the integration
    :param verbose: whether to do some print statements
    :return: tuple containing the time, x, and mu values
    """                     # init; get initial conditions; init
    ksi_s = []              # x indices
    x_s = []                # x values
    mu = []                 # mu values
    x_start, mu_start = initial_conditions(x_center, step_size)
    ksi_s.append(step_size)
    x_s.append(x_start)
    mu.append(mu_start)
                            # set current state and time
    current_state = np.array([x_s[-1], mu[-1]])
    current_ksi = ksi_s[-1]
                            # iterate until x < epsilon
    idx = 0
    refinements = 0
    max_refinements = 25
    while True:
        next_state = RK4(current_state, current_ksi, step_size, derivatives)
                            # when x < epsilon, do the last step over
                            # with a smaller step size max_refinements times
        if next_state[0] < epsilon:
            if refinements < max_refinements:
                step_size /= 2
                refinements += 1
                continue
            else:
                break

        current_ksi += step_size
        ksi_s.append(current_ksi)
        x_s.append(next_state[0])
        mu.append(next_state[1])

        current_state = next_state

        if verbose:         # adjust '1' here for printing frequency
            print('\n\n') if idx == 0 else None
            if idx % 1 == 0: 
                print(f't: {current_ksi}; x: {x[-1]}; mu: {mu[-1]}')
            idx += 1
    
    return np.array(ksi_s), np.array(x_s), np.array(mu)


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


def SM() -> float:
    """ Callable for solar mass

    :return: one solar mass
    """
    return 1.9891 * 10**30


def rho_to_x(
        rho_center: float
    ) -> float:
    """ Converts rho [km/m^3] to its dimensionless counterpart (x)
    
    :param rho_center: central density of the white dwarf
    :return: dimensionless rho
    """
    mu_e = 2                # mean number of nucleons per electron
    return np.cbrt(rho_center / (A() * mu_e))


def x_to_rho(
        x: float
    ) -> float:
    """ Converts dimensionless rho to its real counterpart

    :param x: dimensionless rho
    :return: rho [km/m^3]
    """
    mu_e = 2                # mean number of nucleons per electron
    return A() * mu_e * x**3
            

def step_size_to_ksi(
        r: float,
) -> float:
    """ Converts step size [km] to its dimensionless counterpart
    
    :param r: step size in km
    :return: dimensionless step size
    """
    return r * 1000 / alpha_r()


def ksi_to_step_size(
        ksi: float
) -> float:
    """ Converts dimensionless step size to its real counterpart
    
    :param ksi: dimensionless step size
    :return: step size in km
    """
    return ksi * alpha_r() / 1000


def main():
    initial_values = np.array([[5, 300], [6, 200], [7, 150], [8, 80], [9, 20],
                               [10, 10], [11, 5], [12, 3], [13, 1], [14, 0.5]])
    colours = ('#FFCFEF', '#0A97B0', '#AE445A', '#2A3335', '#F29F58',
               '#AB4459', '#441752', '#355F2E', '#AE445A', '#A27B5C')

    ksi_s_s = []
    x_s = []
    mu_s = []
    final_densities = []
    final_radii = []              
                            # iterate over initial values and store results
    for idx in initial_values:
        ksi_s, x, mu = integrate_white_dwarf(step_size_to_ksi(float(idx[1])),
                                             rho_to_x(10**float(idx[0])),
                                             1e-12)
        ksi_s_s.append(ksi_s)
        x_s.append(x)
        mu_s.append(mu)
        final_radii.append(ksi_to_step_size(ksi_s[-1]))
        final_densities.append(x_to_rho(x[-1]))

                            # plot density as function of the radius
    for ksi_s, x, mu, col, idx in zip(ksi_s_s, x_s, mu_s, colours, initial_values):
        plt.plot(final_radii,
                 final_densities,
                 color = col,
                 label = f'rho_c = 10^{int(idx[0])}')
    plt.xlabel('radius [km]')
    plt.ylabel('density [km/m^3]')
    # plt.ylim(-0.1, plt.ylim()[1])
    plt.legend()
    plt.grid()
    plt.show() 

                            # plot radius in km versus mass in units of solar mass
    for ksi_s, x, mu, col, idx in zip(ksi_s_s, x_s, mu_s, colours, initial_values):
        plt.plot(alpha_m() * mu / SM(),
                 ksi_to_step_size(ksi_s),
                 color = col,
                 label = f'rho_c = 10^{int(idx[0])}')
    plt.xlabel('mass [solar mass]')
    plt.ylabel('radius [km]')
    plt.ylim(bottom = 0)
    plt.legend()
    plt.grid()
    plt.show()    


if __name__ == '__main__':
    main()