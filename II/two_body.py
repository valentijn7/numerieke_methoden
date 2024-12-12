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
        d_c: Dict[str, float]
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
    :return: arrays with x and y positions
    """                     # initialize state and zero-initialized
                            # array for the positions and velocities
    state = np.array([rx0, ry0, vx0, vy0])
    positions = np.zeros((n, 2))
    velocities = np.zeros((n, 2))

    for idx in range(n):    # iterate over n and calculate orbit
        positions[idx] = state[:2]
        velocities[idx] = state[2:]
        state = RK4(state, idx * dt, dt, derivatives, d_c)

    return positions, velocities


def check_when_back(
        orbit: np.array, dt: float = 3600
    ) -> None:
    """
    Checks when a body is back to its starting position

    :param orbit: trajectory
    """
    diff = np.zeros(len(orbit))
    for idx in range(1, len(orbit)):
        diff[idx] = np.linalg.norm(orbit[idx] - orbit[0])
    
    idx_min = (np.argmin(diff[10:]) + 10)
    time_back = (idx_min * dt) / (3600 * 24)
    return time_back


# def calculate_circular_velocity(G: float, M2: float, R: float) -> float:
#     """ Calculates the circular velocity of a body in orbit
#     around another body to stay in a circular orbit

#     :param G: gravitational constant
#     :param M2: mass of the central body
#     :param R: radius of the orbit
#     :return: circular velocity
#     """
#     return np.sqrt(G * M2 / R)


def calculate_and_plot_all(constants: Dict[str, float]) -> None:
    """ Helper function to easily do everything
    for different sets of bodies
    
    :param constants: contains the constants
    """
    rx0, ry0 = constants['R'], 0
    vx0, vy0 = 0, constants['v']
    dt = 3600
    n = int(365.25 * 24) * 5

    r_light, v_light = simulate_light(rx0, ry0, vx0, vy0, dt, n, constants)
    r_heavy, v_heavy = simulate_heavy(r_light, v_light, constants)
    print(f'Back at starting position at {check_when_back(r_light)} days')

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
    relative_variation = np.abs((final_energy - initial_energy) / initial_energy)
    print(f'Initial energy: {initial_energy}')
    print(f'Final energy: {final_energy}')
    print(f'Relative variation: {relative_variation}')
    print()

    plt.figure(figsize = (6, 6))
    plt.plot(energies, color = '#ED8A3F', linewidth = 2)
    plt.xlabel('tijd')
    plt.ylabel('energie')
    plt.show()

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


def main():
    #! TODO's:
    # - maak plotjes voor langere tijdsperiodes met de periode
    # - maak een plotje met tijdsstap versus periode: wordt deze dan accurater?
    # - zoek een verklaring op voor waarom de afwijking bij de aarde het grootste is
    #    -> mogelijk antwoord: probeer andere beginwaarden, zoals die van minor axis
    # - zie de reader voor wat er in de report moet komen

    earth_sun_constants = {
        'G': 6.67430e-11,   # gravitational constant 
        'M1': 5.972e24,     # mass of the Earth
        'M2': 1.989e30,     # mass of the Sun
        'R' : 149.598e9,    # semi-major axis of the Earth
        'v': 29.29e3,       # velocity of the Earth
        'name_1': 'Aarde',   
        'name_2': 'Zon',
    }
    # dt = 3600
    # n = int(365.25 * 24) * 5
    calculate_and_plot_all(earth_sun_constants)#, dt, n)

    moon_earth_constants = {
        'G': 6.67430e-11,   # gravitational constant
        'M1': 0.07346e24,   # mass of the Moon 
        'M2': 5.972e24,     # mass of the Earth
        'R': 0.3844e9,      # semi-major axis of the Moon
        'v': 0.970e3,       # velocity of the Moon
        'name_1': 'Maan',   
        'name_2': 'Aarde',  
    }
    calculate_and_plot_all(moon_earth_constants)

    mercury_sun_constants = {
        'G': 6.67430e-11,   # gravitational constant
        'M1': 0.33010e24,   # mass of the Mercury
        'M2': 1.989e30,     # mass of the Sun
        'R' : 57.909e9,     # semi-major axis of Mercury
        'v' : 38.86e3,      # velocity of Mercury
        'name_1': 'Mercurius',   
        'name_2': 'Zon',  
    }
    calculate_and_plot_all(mercury_sun_constants)



if __name__ == '__main__':
    main()