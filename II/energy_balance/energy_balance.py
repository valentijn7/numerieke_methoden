# numerieke_methoden/II/energy_balance.py

from typing import List, Tuple, Dict, Any
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def RK4(
        current_state: np.array,
        current_time: float,
        step_size: float,
        derivative: callable,
        Q0: float
    ) -> np.array:
    """ Performs one step of the 4th order Runge-Kutta
    
    :param current_state: current state of the system (dependent variable)
    :param current_time: current time of the system (independent variable)
    :param step_size: step size of the integration
    :param derivative: function that returns derivatives for current state and time
    :param Q0: planetary average incoming solar radiation 
    :return: updated values of dependent variable at next time step
    """
                            # calculate the 4 k's
    k1 = step_size * derivative(current_state, current_time, Q0)
    k2 = step_size * derivative(current_state + 0.5 * k1, current_time + 0.5 * step_size, Q0)
    k3 = step_size * derivative(current_state + 0.5 * k2, current_time + 0.5 * step_size, Q0)
    k4 = step_size * derivative(current_state + k3, current_time + step_size, Q0)

                            # calculate and return new state
    return current_state + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def alpha(
        T: float
    ) -> float:
    """ Returns the albedo given a temperature T

    :float T: temperature in K
    :return: albedo coefficient
    """
    alpha_1 = 0.7           # albedo of ice
    alpha_2 = 0.289         # albedo of water
    M = 0.1                 # transition steepness
    T1 = 260                # temps for transition
    T2 = 290
    return alpha_1 + (alpha_2 - alpha_1) * (1 + np.tanh(M * (T - (T1 + T2) / 2))) / 2


def Q(
        x: float,
        Q0: float
    ) -> float:
    """ Returns the the incoming short-wave radiation given an x and Q0

    :param x: variable of integration, x = sin(latitude)
    :param Q0: planetary average incoming solar radiation 
    """
    return Q0 * (1 - 0.241 * (3 * x**2 - 1))


def planetary_avg_T(
        T_values: np.array
    ) -> float:
    """ Returns the planetary average temperature

    :param T_values: T-values
    :return: planetary average temperature
    """
    return np.average(T_values)


def derivative(
        state: np.array,
        x: float,
        Q0: float
    ) -> np.array:
    """ Returns the derivatives dT/dx and dS/sx
    
    :param state: current state (T and S)
    :param x: variable of integration, x = sin(latitude)
    :param Q0: planetary average incoming solar radiation 
    :return: dT/dx and dS/dx
    """
    D = 0.31                # heat-diffusion coefficient
    epsilon = 0.61          # emission coefficient
    sigma = 5.67e-8         # Stefan-Boltzmann coefficient

    T, S = state

    # if T > 1000:            #! TODO vragen of dit prima is om wegshooten te voorkomen
    #     print("large T")
    #     T = 1000
                            # compute dT/dx and dS/dx (while
                            # avoiding division by zero)
    # dTdx = S / max((1 - x**2), 1e-2)
    dTdx = S / ((1 - x**2) + 1e-1)
    dSdx = (epsilon * sigma * T**4 - Q(x, Q0) * (1 - alpha(T))) / D
    # print(T**4)
    return np.array([dTdx, dSdx])


def integrate(
        T0: float,
        Q0: float,
        x_b: float = 1.0,
        n: int = 1000
    ) -> Tuple[np.array, np.array, np.array]:
    """ Integrates from x = 0 to x_b (= 1.0 by default) starting
    with T(0) = T0 and S(0) = 0

    :param T0: starting temperature in K
    :param Q0: planetary average incoming solar radiation
    :param x_b: upper integration bound
    :param n: number of integration steps
    :return: x-, T-, and S-values
    """
    x_values = np.linspace(0, x_b, n + 1)
    dx = x_values[1] - x_values[0]
                            # init state and T & S arrays
    state = np.array([T0, 0.0])
    T_values = np.zeros(n + 1)
    S_values = np.zeros(n + 1)
    T_values[0] = state[0]
    S_values[0] = state[1]
                            # integrate for n steps and save state
    for idx in range(n):
        state = RK4(state, x_values[idx], dx, derivative, Q0)
                            # cap the temperature at 1000K to avoid unrealistic values
        #                     #! TODO vragen of dit prima is om wegshooten te voorkomen
        # if state[0] > 1000:
        #     print('large T')
        #     state[0] = 1000
        # print(state[0], state[1])
                            # save the values
        T_values[idx + 1] = state[0]
        S_values[idx + 1] = state[1]
                            # return all
    return x_values, T_values, S_values


def shoot_T0_for_S1(
        T0: float,
        Q0: float,
    ) -> float:
    """ Wrapper function to shoot a T(0) = T0 for an S(1)
    
    :param T0: initial temp in K
    :param Q0: planetary average incoming solar radiation
    :return: found S(1)
    """
    _, _, S_values = integrate(T0, Q0)
    return S_values[-1]


def newton_shoot(
        T0: float,
        Q0: float,
        dT: float = 1e-2,
        maxiter: int = 100,
        tolerance: float = 1e-6
    ) -> float:
    """ Uses Newton-Raphson to find T0 such that S(1) = 0

    :param T0: initial temp in K
    :param Q0: planetary average incoming solar radiation
    :param maxiter: maximum number of iterations
    :param tolerance: error tolerance
    :return: T0
    """
    for _ in range(maxiter):
                            # get a first candidate for S(1)
                            # and see if it's close enough to 0
        S1_candidate = shoot_T0_for_S1(T0, Q0)
        if np.abs(S1_candidate) < tolerance:
            return T0
                            # if not, update T0 with the derivative
                            # as calculated by the symmetric three-point formula
        # print(1, shoot_T0_for_S1(T0 + dT, Q0))
        # print(2, shoot_T0_for_S1(T0 - dT, Q0))
        
        dS1dT = (shoot_T0_for_S1(T0 + dT, Q0) - \
                 shoot_T0_for_S1(T0 - dT, Q0)) \
                 / (2 * dT)

        # if dS1dT == 0:      # avoid division by zero
        #     print("Warning: division by zero in newton_shoot()")
        #     break

        if np.abs(dS1dT) < 1e-2:

            print("Warning: derivative too small in newton_shoot()")
            break
        
        print(f"Iteration {_}: T0 = {T0:.3f}, S1 = {S1_candidate:.3f}, dS1/dT = {dS1dT:.3e}")
        
        if np.isnan(S1_candidate) or np.isnan(dS1dT):
            print("Warning: newton_shoot() found a nan")
            break

                            # see if the Newton-Raphson method converges
        T0_next = T0 - S1_candidate / dS1dT
        if np.abs(T0_next - T0) < tolerance:
            return T0_next
        T0 = T0_next        # continue with the next iteration

    print("Warning: newton_shoot() did not converge")
    return T0



def main():
    T0_1 = 215
    T0_2 = 225
    T0_3 = 235
    Q0_1 = 300
    Q0_2 = 350
    Q0_3 = 400
    x_values_1, T_values_1, S_values_1 = integrate(T0_1, Q0_1)
    T_avg_1 = planetary_avg_T(T_values_1)
    x_values_2, T_values_2, S_values_2 = integrate(T0_2, Q0_2)
    T_avg_2 = planetary_avg_T(T_values_2)
    x_values_3, T_values_3, S_values_3 = integrate(T0_3, Q0_3)
    T_avg_3 = planetary_avg_T(T_values_3)

                            # plot the results
                            #! TODO zet de T waardes er ook in in label
    plt.figure(figsize = (6, 6))
    plt.plot(x_values_1, T_values_1, label = r'$\mathrm{T}, \mathrm{T} =, \mathrm{Q}_0 = 300$',
             color = '#FF0000')
    plt.axhline(T_avg_1, color = '#FF0000', linestyle = 'dashed',
                label = r'$\mathrm{T}_{\mathrm{avg}}, \mathrm{Q}_0 = 300$')
    plt.plot(x_values_2, T_values_2, label = r'$\mathrm{T}, \mathrm{Q}_0 = 350$',
             color = '#00FF00')
    plt.axhline(T_avg_2, color = '#00FF00', linestyle = 'dashed',
                label = r'$\mathrm{T}_{\mathrm{avg}}, \mathrm{Q}_0 = 350$')
    plt.plot(x_values_3, T_values_3, label = r'$\mathrm{T}, \mathrm{Q}_0 = 400$',
             color = '#0000FF')
    plt.axhline(T_avg_3, color = '#0000FF', linestyle = 'dashed',
                label = r'$\mathrm{T}_{\mathrm{avg}}, \mathrm{Q}_0 = 400$')


    plt.xlabel('x (integratievariabele)', fontsize = 14)
    plt.ylabel('T (Kelvin)', fontsize = 14)
    plt.legend()
    plt.grid(True, which = 'both', alpha = 0.8)
    plt.show()


    # T0_values = np.linspace(230, 350, 100)
    # Q0 = 300
    # S1_values = [0] * len(T0_values)
    # for idx in range(len(T0_values)):
    #     S1_values[idx] = shoot_T0_for_S1(T0_values[idx], Q0)

    # plt.figure(figsize = (6, 6))
    # plt.plot(T0_values, S1_values)
    # plt.xlabel('T0')
    # plt.ylabel('S1')
    # plt.yscale('log')
    # # plt.legend()
    # plt.grid(True, which = 'both', alpha = 0.8)
    # plt.show()

    # T0 = 300
    # Q0 = 300
    # T0 = newton_shoot(T0, Q0)
    # print(f'Found T0 of {T0:.3f}')
    # print(f'Verify: S(1) = {shoot_T0_for_S1(T0, Q0):.3f}')


    Q0_values = np.linspace(200, 500, 10)
    Tav_values = np.zeros(len(Q0_values))
    T_profiles = np.zeros(len(Q0_values))
  
    T0 = 230
    idx = 0
    for Q0 in Q0_values:
        print(f'\nT0: {T0}, Q0: {Q0}')
        T0 = newton_shoot(T0, Q0)
        x_values, T_values, _ = integrate(T0, Q0)
        Tav_values[idx] = planetary_avg_T(x_values, T_values)
        T_profiles[idx] = T_values[-1]
        print(f'Found T_avg of {Tav_values[idx]:.3f} for Q0 = {Q0}')
        idx += 1

    # plt.figure(figsize = (6, 6))
    # plt.plot(Q0_values, Tav_values, label = 'Tav')
    # plt.plot(Q0_values, T_profiles, label = 'T(1)')
    # plt.xlabel('Q0')
    # plt.ylabel('T')
    # plt.legend()
    # plt.grid(True, which = 'both', alpha = 0.8)
    # plt.show()

    print('\n\nTake 2\n\n')

    Q0_values_2 = np.linspace(500, 200, 10)
    Tav_values_2 = np.zeros(len(Q0_values))
    T_profiles_2 = np.zeros(len(Q0_values))
  
    T0 = 400
    idx = 0
    for Q0 in Q0_values_2:
        print(f'\nT0: {T0}, Q0: {Q0}')
        T0 = newton_shoot(T0, Q0)
        x_values, T_values, _ = integrate(T0, Q0)
        Tav_values_2[idx] = planetary_avg_T(T_values)
        T_profiles_2[idx] = T_values[-1]
        print(f'Found T_avg of {Tav_values_2[idx]:.3f} for Q0 = {Q0}')
        idx += 1

    plt.figure(figsize = (6, 6))
    plt.plot(Q0_values, Tav_values, label = 'Tav', linewidth = 5)
    plt.plot(Q0_values, T_profiles, label = 'T(1)', linewidth = 5)
    plt.plot(Q0_values_2, Tav_values_2, label = 'Tav_2')
    plt.plot(Q0_values_2, T_profiles_2, label = 'T(1)_2')
    plt.xlabel('Q0')
    plt.ylabel('T')
    plt.legend()
    plt.grid(True, which = 'both', alpha = 0.8)
    plt.show()


if __name__ == '__main__':
    main()