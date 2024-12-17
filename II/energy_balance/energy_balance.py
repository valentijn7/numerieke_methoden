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
                            # compute dT/dx and dS/dx (while
                            # avoiding division by zero)
    dTdx = S / max((1 - x**2), 1e-10)
    dSdx = (epsilon * sigma * T**4 - Q(x, Q0) * (1 - alpha(T))) / D

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
        T_values[idx + 1] = state[0]
        S_values[idx + 1] = state[1]
                            # return all
    return x_values, T_values, S_values


# def f(
#         x: float,
#         roots: List[float]
#     ) -> float:
#     """ Returns f(x)

#     :param x: evaluate f at x
#     :param roots: roots (with only x = 1 for S(1))
#     :return: f(x)
#     """
#     return np.prod([x - root for root in roots])


# def df(
#         x: float,
#         roots: List[float],
#         dx: float = 1e-6
#     ) -> float:
#     """ Returns df/dx where f is a polynomial defined as
    
#     f(x) = (r1 - x) * (r2 - x) * ... * (rn - x)

#     where r1, r2, ..., rn are the roots of the polynomial.
#     To obtain f'(x), we use the 'symmetric three-point formula'

#     :param x: point of evaluation
#     :param roots: roots (with only x = 1 for S(1))
#     :param dx: step size
#     :return: df/dx
#     """
#     return (f(x + dx, roots) - f(x - dx, roots)) / (2 * dx)


# def newton_raphson(
#         roots: List[float],
#         x0: float,
#         maxiter: int = 1000,
#         dec: float = 6
#     ) -> Tuple[float, float, int]:
#     """
#     Determines the root of the equation f(x) = 0 within [a, b]
#     using the Newton-Raphson method

#     :param roots: list of polynomial roots
#     :param x0: double with the initial guess
#     :param maxiter: maximum iterations (to prevent infinite loops)
#     :param dec: number of decimals to round the result to
#     :return: a tuple (Root, Error, Number of Iterations)
#     """
#     n = 0
#     tolerance = 1.0e-6
#     while n < maxiter:
#         fx = f(x0, roots)       # calculate f(x0)
#         dfx = df(x0, roots)     # calculate f'(x0)
        
#         x1 = x0 - fx / dfx 
#                                 # if within the error tolerance, return result
#         if abs(x1 - x0) < tolerance:
#             return x1, abs(x1 - x0), n, dec
        
#         if n + 1 != maxiter:    # if not at the last iteration, update x0
#             x0 = x1              
#         n += 1
#                                 # if the loop exits, the method did not converge
#     return x1, abs(x1 - x0), n


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
        dT: float = 0.1,
        maxiter: int = 1000,
        tolerance: float = 1e-6
    ) -> float:
    """ Uses Newton-Raphson to find T0 such that S(1) = 0

    :param T0: initial temp in K
    :param Q0: planetary average incoming solar radiation
    :param maxiter: maximum number of iterations
    :param tolerance: error tolerance
    :return: T0
    """
    for idx in range(maxiter):
                            # get a first candidate for S(1)
                            # and see if it's close enough to 0
        S1_candidate = shoot_T0_for_S1(T0, Q0)
        if np.abs(S1_candidate) < tolerance:
            return T0
                            # if not, update T0 with the derivative
                            # as calculated by the symmetric three-point formula
        dS1dT = (shoot_T0_for_S1(T0 + dT, Q0) - \
                 shoot_T0_for_S1(T0 - dT, Q0)) \
                 / 2 * dT

        if dS1dT == 0:      # avoid division by zero
            print("Warning: division by zero in newton_shoot()")
            break
                            # see if the Newton-Raphson method converges
        T0_next = T0 - S1_candidate / dS1dT
        if np.abs(T0_next - T0) < tolerance:
            return T0_next
        T0 = T0_next        # continue with the next iteration

    print("Warning: newton_shoot() did not converge")
    return T0



def main():
    T0 = 280
    Q0 = 300
    x_values, T_values, S_values = integrate(T0, Q0)

    print(f'Found S(1) of {S_values[-1]:.3f}')
    print(f'Found T(1) of {T_values[-1]:.3f}')

                            # plot the results
    plt.figure(figsize = (6, 6))
    plt.plot(x_values, T_values, label = 'T')
    plt.plot(x_values, S_values, label = 'S')
    plt.xlabel('x')
    plt.ylabel('T, S')
    plt.legend()
    plt.grid(True, which = 'both', alpha = 0.8)
    plt.show()


    T0_values = np.linspace(250, 350, 100)
    Q0 = 300
    S1_values = [0] * len(T0_values)
    for idx in range(len(T0_values)):
        S1_values[idx] = shoot_T0_for_S1(T0_values[idx], Q0)

    plt.figure(figsize = (6, 6))
    plt.plot(T0_values, S1_values)
    plt.xlabel('T0')
    plt.ylabel('S1')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which = 'both', alpha = 0.8)
    plt.show()


    T0 = newton_shoot(280, 300)
    print(f'Found T0 of {T0:.3f}')
    print(f'Verify: S(1) = {shoot_T0_for_S1(T0, Q0):.3f}')


if __name__ == '__main__':
    main()