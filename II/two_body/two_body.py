# numerieke_methoden/II/two_body.py

from typing import Tuple, Dict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams['axes.formatter.useoffset'] = False


def RK4(
        current_state: np.array,
        current_time: float,
        step_size: float,
        derivative: callable,
        d_c: Dict[str, float]
    ) -> np.array:
    """ Performs one step of the 4th order Runge-Kutta
    
    :param current_state: current state of the system (dependent variable)
    :param current_time: current time of the system (independent variable)
    :param step_size: step size of the integration
    :param derivative: function that returns derivatives for current state and time
    :param d_c: dictionary with constants
    :return: updated values of dependent variable at next time step
    """
                            # calculate the 4 k's
    k1 = step_size * derivative(current_state, current_time, d_c)
    k2 = step_size * derivative(current_state + 0.5 * k1, current_time + 0.5 * step_size, d_c)
    k3 = step_size * derivative(current_state + 0.5 * k2, current_time + 0.5 * step_size, d_c)
    k4 = step_size * derivative(current_state + k3, current_time + step_size, d_c)

                            # calculate and return new state
    return current_state + (k1 + 2 * k2 + 2 * k3 + k4) / 6


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


def velocity_verlet(
        current_state: np.array,
        current_time: float,
        step_size: float,
        derivative: callable,
        d_c: Dict[str, float]
    ) -> np.array:
    """ Performs one step of the Velocity Verlet method
    
    :param current_state: current state of the system (dependent variable)
    :param current_time: current time of the system (independent variable)
    :param step_size: step size of the integration
    :param derivative: function that returns derivatives for current state and time
    :param d_c: dictionary with constants
    :return: updated values of dependent variable at next time step
    """                     
    acceleration = derivative(current_state, current_time, d_c)[2:]
                            # extract position and state and perform VV-step
    position = current_state[:2]
    velocity = current_state[2:]
    new_position = position + velocity * step_size + 0.5 * acceleration * step_size**2
    new_acceleration = derivative(np.concatenate((new_position, velocity)),
                                  current_time + step_size, d_c)[2:]
    new_velocity = velocity + 0.5 * (acceleration + new_acceleration) * step_size
    return np.concatenate((new_position, new_velocity))


def derivatives(
        state: np.array,
        t: float,
        d_c: Dict[str, float]
    ) -> np.array:
    """ Specific to the two-body problem, using the
    equations (26) and (29) from the reader

    :param state: current state of the system
    :param t: current time of the system
    :param d_c: dictionary with constants
    :return: derivatives of the dependent variables
    """
    _ = t
    G, M1, M2 = d_c['G'], d_c['M1'], d_c['M2']
    rx, ry = state[:2]
    vx, vy = state[2:]
    r_cubed = (rx**2 + ry**2)**(3/2)

    drx_dt = vx
    dry_dt = vy

    factor = -(G * M2 / (1 + M1 / M2)**2)
    dvx_dt = factor * (rx / r_cubed)
    dvy_dt = factor * (ry / r_cubed)

    return np.array([drx_dt, dry_dt, dvx_dt, dvy_dt])


def f_T(M1: float, vx: float, vy: float) -> float:
    """ Calculates kinetic energy of a body in orbit
    
    :param M1: mass of the body
    :param vx: x velocity
    :param vy: y velocity
    :return: kinetic energy
    """
    return 0.5 * M1 * (vx**2 + vy**2)


def f_V(G: float, M1: float, M2: float, rx: float, ry: float) -> float:
    """ Calculates potential energy of a body in orbit
    
    :param G: gravitational constant
    :param M1: mass of the body
    :param M2: mass of the central body
    :param rx: x position
    :param ry: y position
    :return: potential energy
    """
    return -G * M1 * M2 / np.sqrt(rx**2 + ry**2)


def calculate_energy(
        rx_1: float,
        ry_1: float,
        vx_1: float,
        vy_1: float,
        rx_2: float,
        ry_2: float,
        vx_2: float,
        vy_2: float,
        d_c: Dict[str, float]
) -> Tuple[float, np.array, np.array]:
    """ Calculates the total energy of a body in orbit
    around another body

    :param rx_1: x position 1st body
    :param ry_1: y position 2nd body
    :param vx_1: x velocity 1st body
    :param vy_1: y velocity 2nd body
    :param rx_2: x position 2nd body
    :param ry_2: y position 2nd body
    :param vx_2: x velocity 2nd body
    :param vy_2: y velocity 2nd body
    :param d_c: dictionary with constants
    :return: total energy
    """
    G, M1, M2 = d_c['G'], d_c['M1'], d_c['M2']
    T_1 = f_T(M1, vx_1, vy_1)
    T_2 = f_T(M2, vx_2, vy_2)
    V = f_V(G, M1, M2, np.abs(rx_2 - rx_1), np.abs(ry_2 - ry_1))
    return (T_1 + T_2) + V


def simulate_heavy(
        position: np.array,
        velocity: np.array,
        d_c: Dict[str, float]
    ) -> Tuple[float, float]:
    """ Calculates the position and velocity of the heavy body
    
    :param position: position of the light body
    :param velocity: velocity of the light body
    :param d_c: dictionary with constants
    :return: position and velocity of the heavy body
    """
    M1, M2 = d_c['M1'], d_c['M2']
    return -M1 / M2 * position, -M1 / M2 * velocity


def simulate_light(
        rx0: float,
        ry0: float,
        vx0: float,
        vy0: float,
        dt: float,
        n: int,
        d_c: Dict[str, float],
        method: str = 'RK4'
    ):
    """ Simulates the orbit of a body around another body
    from a starting position (rx0, ry0) and velocity
    (vx0, vy0) for n steps of size dt

    :param rx0: initial x position
    :param ry0: initial y position
    :param vx0: initial x velocity
    :param vy0: initial y velocity
    :param dt: step size
    :param n: number of steps
    :param d_c: dictionary with constants
    :param method: which method to use: RK4, FE (forward Euler), VV (velocity Verlet)
    :return: arrays with x and y positions
    """                     
    if method == 'RK4':
        f = RK4
    elif method == 'FE':
        f = forward_euler
    elif method == 'VV':
        f = velocity_verlet
    else:
        raise ValueError('Invalid method')
                            # initialize state and zero-initialized
                            # array for the positions and velocities
    state = np.array([rx0, ry0, vx0, vy0])
    positions = np.zeros((n, 2))
    velocities = np.zeros((n, 2))

    for idx in range(n):    # iterate over n and calculate orbit
        positions[idx] = state[:2]
        velocities[idx] = state[2:]
        state = f(state, idx * dt, dt, derivatives, d_c)

    return positions, velocities


def check_when_back(
        distances: np.array,
        times: np.array
    ) -> float:
    """
    Checks when a body is back to its starting position

    :param distances: distances between light and heavy body
    :param times: time array
    :param dt: step size
    :return: time when the body is back at starting position
    """
    indices = []            # fina maximal distances, assuming two points
                            # don't have the exact same distance (unlikely)
    for idx in range(1, len(distances) - 1):
        if distances[idx] > distances[idx - 1] and \
            distances[idx] > distances[idx + 1]:
            indices.append(idx)

    if len(indices) < 2:    # if there are not enough maxima, return 0
        print('Not enough maxima found to determine period')
        return 0
                            # return the period in days
    return (times[indices[1]] - times[indices[0]])


# def calculate_circular_velocity(
#         G: float,
#         M2: float,
#         R: float
#     ) -> float:
#     """ Calculates the circular velocity of a body in orbit
#     around another body to stay in a circular orbit

#     :param G: gravitational constant
#     :param M2: mass of the central body
#     :param R: radius of the orbit
#     :return: circular velocity
#     """
#     return np.sqrt(G * M2 / R)


def Kepler_period(
        d_c: Dict[str, float]
    ) -> float:
    """ Calculates the period of a body in orbit around another
    body using Kepler's third law

    :param d_c: dictionary with constants
    :return: period
    """
    G, M1, M2, R = d_c['G'], d_c['M1'], d_c['M2'], d_c['R']
    return 2 * np.pi * np.sqrt(R**3 / (G * (M1 + M2))) 


def simulate_and_get_period(
        dt: int,
        constants: Dict[str, float],
        total_years: int = 3
    ) -> float:
    """ Helper function to get the period for multiple dt sizes

    :param dt: step size
    :param constants: contains the constants
    :return: period
    """
    rx0, ry0 = constants['R'], 0
    vx0, vy0 = 0, constants['v']
    n = int(365.25 * 24) * total_years

    r_light, _ = simulate_light(rx0, ry0, vx0, vy0, dt, n, constants)
    r_heavy, _ = simulate_heavy(r_light, _, constants)
    distances = np.linalg.norm(r_light - r_heavy, axis = 1)
    times = np.arange(n) * dt / (3600 * 24)
    return check_when_back(distances, times)


def calculate_and_plot_all(
        constants: Dict[str, float],
        dt: int, total_years: int
    ) -> None:
    """ Helper function to easily do everything
    for different sets of bodies
    
    :param constants: contains the constants
    :param dt: step size
    :param total_years: total years to simulate
    :return: Tuple with multiple arrays and/or labels for combined plotting
    """
    rx0, ry0 = constants['R'], 0
    vx0, vy0 = 0, constants['v']
    n = int(365.25 * 24) * total_years

    r_light, v_light = simulate_light(rx0, ry0, vx0, vy0, dt, n, constants)
    r_heavy, v_heavy = simulate_heavy(r_light, v_light, constants)

    distances = np.linalg.norm(r_light - r_heavy, axis = 1)
    times = np.arange(n) * dt / (3600 * 24) # in days

    period = check_when_back(distances, times)
    print(f'Back at starting position at {period:.2f} days')

    # Commented-out and moved to main() for combined plotting
    # plt.figure(figsize = (6, 6))
    # plt.plot(times, distances, color = '#ED8A3F', linewidth = 2)
    # plt.xlabel('tijd (dagen)')
    # plt.ylabel('afstand (m)')
    # plt.grid()
    # plt.show()
    #! TODO voeg toe dat final energy de energie is aan het einde van de baan, niet de simulatie

    initial_energy = calculate_energy(r_light[0, 0], r_light[0, 1],
                                      v_light[0, 0], v_light[0, 1],
                                      r_heavy[0, 0], r_heavy[0, 1],
                                      v_heavy[0, 0], v_heavy[0, 1],
                                      constants)
    final_energy = calculate_energy(r_light[-1, 0], r_light[-1, 1],
                                    v_light[-1, 0], v_light[-1, 1],
                                    r_heavy[-1, 0], r_heavy[-1, 1],
                                    v_heavy[-1, 0], v_heavy[-1, 1],
                                    constants)
    energies = np.zeros(n)
    for idx in range(n):
        energies[idx] = calculate_energy(r_light[idx, 0], r_light[idx, 1],
                                       v_light[idx, 0], v_light[idx, 1],
                                       r_heavy[idx, 0], r_heavy[idx, 1],
                                       v_heavy[idx, 0], v_heavy[idx, 1],
                                       constants)
    relative_change = np.abs((final_energy - initial_energy) / initial_energy) * 100
    print(f'Initial energy: {initial_energy}')
    print(f'10^-8 of initial energy: {initial_energy * 10**-8}')
    print(f'Final energy: {final_energy}')
    print(f'Relative change: {relative_change}%')
    ### Print the percentage 
    print('\t-----')
    print(f'Obtained period: {period:.2f} days')
    print(f'Theoretical period: {Kepler_period(constants) / (3600 * 24):.2f} days')
    print()
    print()

    # Uncomment to show the approximate energy conservation for each pair
    plt.figure(figsize = (6, 6))
    plt.plot(energies, color = '#ED8A3F', linewidth = 2)
    plt.xlabel('tijd')
    plt.ylabel('energie')
    plt.show()

    # Uncomment to show the orbits of the two bodies
    plt.figure(figsize = (6, 6))
    plt.plot(r_light[:, 0], r_light[:, 1], color = 'black', linewidth = 2,
             label = f'Baan van de {constants["name_1"]}')
    plt.plot(r_heavy[:, 0], r_heavy[:, 1], color = 'red', linewidth = 5,
             label = f'Baan van de {constants["name_2"]}')
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

    return distances, times, f'{constants["name_1"]} - {constants["name_2"]}'


def main():
    ### CHANGE INITIAL CONDITIONS HERE!
    # Source: https://nssdc.gsfc.nasa.gov/planetary/factsheet/

    earth_sun_constants = {
        'G': 6.67430e-11,   # gravitational constant 
        'M1': 5.972e24,     # mass of the Earth
        'M2': 1.989e30,     # mass of the Sun
        'R' : 149.598e9,    # semi-major axis of the Earth
        'v': 29.29e3,       # velocity of the Earth
        'name_1': 'Aarde',   
        'name_2': 'Zon',
    }
    moon_earth_constants = {
        'G': 6.67430e-11,   # gravitational constant
        'M1': 0.07346e24,   # mass of the Moon 
        'M2': 5.972e24,     # mass of the Earth
        'R': 0.3844e9,      # semi-major axis of the Moon
        'v': 0.970e3,       # velocity of the Moon
        'name_1': 'Maan',   
        'name_2': 'Aarde',  
    }
    mercury_sun_constants = {
        'G': 6.67430e-11,   # gravitational constant
        'M1': 0.33010e24,   # mass of the Mercury
        'M2': 1.989e30,     # mass of the Sun
        'R' : 57.909e9,     # semi-major axis of Mercury
        'v' : 38.86e3,      # velocity of Mercury
        'name_1': 'Mercurius',   
        'name_2': 'Zon',  
    }
    mercury_sun_constants_high_velocity = {
        'G': 6.67430e-11,   # gravitational constant
        'M1': 0.33010e24,   # mass of the Mercury
        'M2': 1.989e30,     # mass of the Sun
        'R' : 57.909e9,     # semi-major axis of Mercury
        'v' : 47.9e3,       # velocity of Mercury now 47.9e3 to see get a more
                            # circular orbit
        'name_1': 'Mercurius',   
        'name_2': 'Zon',  
    }

    dt = 3600
    ty = 2                  # total years

                            # Approximated periods:
                            # - Sun-Earth: 347.75 days
                            # - Earth-Moon: 25.13 days
                            # - Sun-Mercury: 56.62 days

                            # catch the distances, times, and
                            # labels for combined plotting
    d1, t1, l1 = calculate_and_plot_all(earth_sun_constants, dt, ty)
    d2, t2, l2 = calculate_and_plot_all(moon_earth_constants, dt, ty)
    d3, t3, l3 = calculate_and_plot_all(mercury_sun_constants, dt, ty) 
    d3_2, t3_2, l3_2 = calculate_and_plot_all(mercury_sun_constants_high_velocity, dt, ty) 

    plt.figure(figsize = (6, 6))
    plt.plot(t2, (d2 - np.mean(d2)) / np.std(d2), color = '#FF0000', linewidth = 1.5, label = l2 + ' (z-score)')
    plt.plot(t3, (d3 - np.mean(d3)) / np.std(d3), color = '#00FF00', linewidth = 1.5, label = l3 + ' (z-score)')
    plt.plot(t3_2, (d3_2 - np.mean(d3_2)) / np.std(d3_2), '#4C3575', linewidth = 1.5,
             label = l3_2 + ' ($v_{{start}} = 48.9e3$) (z-score)')
    plt.plot(t1, (d1 - np.mean(d1)) / np.std(d1), color = '#0000FF', linewidth = 1.5, label = l1 + ' (z-score)')
    # add vertical lines where one period of 347.75 days is reached
    for idx in range(1, ty + 1):
        plt.axvline(x = 347.75 * idx, color = '#0000FF', linestyle = '--', linewidth = 1.5)
    plt.plot([], [], color = '#0000FF', linestyle = '--', linewidth = 1.5, label = 'Volledige omwenteling Aarde-Zon')

    plt.xlabel('tijd (dagen)', fontsize = 14)
    plt.ylabel('afstand (m), gestandaardiseerd', fontsize = 14)
    plt.legend(fontsize = 14, facecolor = '#F0F0F0')
    plt.grid(True, which = 'both', alpha = 0.8)
    plt.tight_layout()
    plt.show()

    # now, we want to plot the period versus different starting speeds of
    # the Mercury-Sun system
    theo_period = Kepler_period(mercury_sun_constants) / (3600 * 24)
    print(f'Theoretical period for Mercury-Sun: {theo_period:.2f} days')
    speeds = np.linspace(38.86e3, 38.86e3 + 10e3, 50)
    periods = []
    for speed in speeds:
        mercury_sun_constants['v'] = speed
        periods.append(simulate_and_get_period(dt, mercury_sun_constants, ty))
        print(f'Period for speed = {speed:.2f}: {periods[-1]:.2f} days')

    plt.figure(figsize = (6, 6))
    plt.plot(speeds, periods, color = '#BB0029', linewidth = 2,
             label = 'Gesimuleerde periode')
    plt.axhline(y = theo_period, color = '#000000', linestyle = '--', linewidth = 1.5,
                label = '3e wet van Kepler')
    plt.xlabel('beginsnelheid (m/s)', fontsize = 14)
    plt.ylabel('periode (dagen)', fontsize = 14)
    plt.grid(True, which = 'both', alpha = 0.8)
    plt.legend(fontsize = 14, facecolor = '#F0F0F0')
    plt.tight_layout()
    plt.show()

    #################################################################
    # extra task: check for energy conservation for different methods
    ty = 2
    dt = 3600
    constants = earth_sun_constants

    rx0, ry0 = constants['R'], 0
    vx0, vy0 = 0, constants['v']
    n = int(365.25 * 24) * ty
                                # Calculate relative change for RK4
    r_light, v_light = simulate_light(rx0, ry0, vx0, vy0, dt, n, constants, 'RK4')
    r_heavy, v_heavy = simulate_heavy(r_light, v_light, constants)

    initial_energy = calculate_energy(r_light[0, 0], r_light[0, 1],
                                      v_light[0, 0], v_light[0, 1],
                                      r_heavy[0, 0], r_heavy[0, 1],
                                      v_heavy[0, 0], v_heavy[0, 1],
                                      constants)
    final_energy = calculate_energy(r_light[-1, 0], r_light[-1, 1],
                                    v_light[-1, 0], v_light[-1, 1],
                                    r_heavy[-1, 0], r_heavy[-1, 1],
                                    v_heavy[-1, 0], v_heavy[-1, 1],
                                    constants)
    relative_change = np.abs((final_energy - initial_energy) / initial_energy) * 100
    print(f'Relative change for RK4, dt = 3600: {relative_change}%')
                                # Calculate relative change for FE
    r_light, v_light = simulate_light(rx0, ry0, vx0, vy0, dt, n, constants, 'FE')
    initial_energy = calculate_energy(r_light[0, 0], r_light[0, 1],
                                      v_light[0, 0], v_light[0, 1],
                                      r_heavy[0, 0], r_heavy[0, 1],
                                      v_heavy[0, 0], v_heavy[0, 1],
                                      constants)
    final_energy = calculate_energy(r_light[-1, 0], r_light[-1, 1],
                                    v_light[-1, 0], v_light[-1, 1],
                                    r_heavy[-1, 0], r_heavy[-1, 1],
                                    v_heavy[-1, 0], v_heavy[-1, 1],
                                    constants)
    relative_change = np.abs((final_energy - initial_energy) / initial_energy) * 100
    print(f'Relative change for FE, dt = 3600: {relative_change}%')
                                # Calculate relative change for VV
    r_light, v_light = simulate_light(rx0, ry0, vx0, vy0, dt, n, constants, 'VV')
    initial_energy = calculate_energy(r_light[0, 0], r_light[0, 1],
                                      v_light[0, 0], v_light[0, 1],
                                      r_heavy[0, 0], r_heavy[0, 1],
                                      v_heavy[0, 0], v_heavy[0, 1],
                                      constants)
    final_energy = calculate_energy(r_light[-1, 0], r_light[-1, 1],
                                    v_light[-1, 0], v_light[-1, 1],
                                    r_heavy[-1, 0], r_heavy[-1, 1],
                                    v_heavy[-1, 0], v_heavy[-1, 1],
                                    constants)
    relative_change = np.abs((final_energy - initial_energy) / initial_energy) * 100
    print(f'Relative change for VV, dt = 3600: {relative_change}%')
                                # Calculate relative change for VV with dt = 900
    dt = 900
    r_light, v_light = simulate_light(rx0, ry0, vx0, vy0, dt, n, constants, 'VV')
    initial_energy = calculate_energy(r_light[0, 0], r_light[0, 1],
                                      v_light[0, 0], v_light[0, 1],
                                      r_heavy[0, 0], r_heavy[0, 1],
                                      v_heavy[0, 0], v_heavy[0, 1],
                                      constants)
    final_energy = calculate_energy(r_light[-1, 0], r_light[-1, 1],
                                    v_light[-1, 0], v_light[-1, 1],
                                    r_heavy[-1, 0], r_heavy[-1, 1],
                                    v_heavy[-1, 0], v_heavy[-1, 1],
                                    constants)
    relative_change = np.abs((final_energy - initial_energy) / initial_energy) * 100
    print(f'Relative change for VV, dt = 900: {relative_change}%')


    # lastly, uncomment the following code to see the time period does not make
    # the biggest difference in the accuracy of the simulation, apart from 
    # instability kicking in at a certain point
    # ty = 10
    # dt_sizes = [7200, 3600, 1800, 900, 800, 700, 600, 500]
    # periods = []
    # for dt in dt_sizes:
    #     periods.append(simulate_and_get_period(dt, earth_sun_constants, ty))
    #     print(f'Period for dt = {dt}: {periods[-1]:.2f} days')


if __name__ == '__main__':
    main()