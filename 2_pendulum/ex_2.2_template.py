#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ######################### PART 2 ##########################
"""
Still considering the simple pendulum of length L, we can now try to integrate
numerically the equations of motion.
For instance, we can use the Euler algorithm.
We can then compare the results of the numerical integration with the small
angles approximations (dashed), seeing how long it does keep.
For instance, the equation of motion that we want to integrate is
    theta''(t) = -(g/L) * sin(theta(t))
which can be decomposed in
    omega'(t) = -(g/L) * sin(theta(t))
    theta'(t) = omega(t)
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
plt.close('all')

n_frames = 10001
t = np.linspace(0, 100, n_frames)

dt = t[1] - t[0]  # timestep

g = 9.8  # m/s**2
L = 1.
theta_0 = np.deg2rad(60)
omega_0 = 0


# Define the derivatives of theta and omega (the function to integrate)
def derivatives_simple_pendulum(state: np.array, t: float) -> np.array:
    """
        state: numpy array giving the state of the pendulum at time t
        (theta and omega)

        Function that returns the derivatives
            theta'(t) = omega(t)
            omega'(t) = ...

        Returns a np.array containing the derivatives (theta', omega')
    """
    theta = state[0]
    omega = state[1]
    theta_prime = omega
    omega_prime = - (g / L) * np.sin(theta)
    return np.array([theta_prime, omega_prime])



def euler(old_state:np.array, t: float, dt: float, derivatives) -> np.array:
    """
        old_state: numpy array giving the state of the pendulum at time t
        t: starting time
        dt: integration step
        derivatives: function that calculate the derivatives of the coordinates

        Function that performs an integration step using the euler
        algorithm

        Returns an np.array containing the new state (theta, omega)
    """
    return old_state + dt * derivatives(old_state, t)


def pendulum_location(theta, L):
    """
    Returns the location of the pendulum end as a function of the angle and
    length
    """
    x = L * np.sin(theta)
    y = - L * np.cos(theta)
    return np.array([x, y])


# create arrays containing the initial angle/velocity/location of the pendulum
state = np.array([[theta_0, omega_0]])
x_y = pendulum_location(theta_0, L)

state_sa = np.array([[theta_0, omega_0]])
x_y_sa = pendulum_location(theta_0, L)


# update the pendulum angle/velocity/location using the euler integration
for index, time in enumerate(t[1:]):
    state_new = euler(state[index, :], time, dt,
                            derivatives_simple_pendulum)
    x_y_new = pendulum_location(state_new[0], L)

    state = np.vstack((state, state_new))
    x_y = np.vstack((x_y, x_y_new))


def execute_part2():
    t_window = 0.1*t[-1]

    fig, ax = plt.subplots(2, 2)

    anim_theta,         = ax[0, 0].plot([], [], 'b-')
    anim_theta_sa,      = ax[0, 0].plot([], [], 'b--')
    ax[0, 0].set_xlim(0, t_window)
    ax[0, 0].set_ylim(-1.1*np.max(np.abs(state[:, 0])),
                      1.1*np.max(np.abs(state[:, 0])))
    ax[0, 0].set_xlabel('t')
    ax[0, 0].set_ylabel(r'$\theta$(t)')

    anim_omega,         = ax[0, 1].plot([], [], 'r-')
    anim_omega_sa,      = ax[0, 1].plot([], [], 'r--')
    ax[0, 1].set_xlim(0, t_window)
    ax[0, 1].set_ylim(-1.1*np.max(np.abs(state[:, 1])),
                      1.1*np.max(np.abs(state[:, 1])))
    ax[0, 1].set_xlabel('t')
    ax[0, 1].set_ylabel(r'$\omega$(t)')

    anim_pendulum,      = ax[1, 0].plot([], [], 'bo-')
    anim_trajectory,    = ax[1, 0].plot([], [], 'r-')
    ax[1, 0].set_xlim(-1.1*np.max(np.abs(x_y[:, 0])),
                      1.1*np.max(np.abs(x_y[:, 0])))
    ax[1, 0].set_ylim(-1.1*np.max(np.abs(x_y[:, 1])),
                      1.1*np.max(np.abs(x_y[:, 1])))
    ax[1, 0].set_xlabel('x(t)')
    ax[1, 0].set_ylabel('y(t)')

    anim_phase,         = ax[1, 1].plot([], [], 'b-')
    ax[1, 1].set_xlim(-1.1*np.max(np.abs(state[:, 0])),
                      1.1*np.max(np.abs(state[:, 0])))
    ax[1, 1].set_ylim(-1.1*np.max(np.abs(state[:, 1])),
                      1.1*np.max(np.abs(state[:, 1])))
    ax[1, 1].set_xlabel(r'$\theta$(t)')
    ax[1, 1].set_ylabel(r'$\omega$(t)')

    def init():
        anim_theta.set_data([], [])
        anim_theta_sa.set_data([], [])
        anim_omega.set_data([], [])
        anim_omega_sa.set_data([], [])
        anim_pendulum.set_data([], [])
        anim_trajectory.set_data([], [])
        anim_phase.set_data([], [])

    # The animation function which is called every frame
    def animate(i, state, state_sa, x_y, x_y_sa):

        anim_theta.set_data(t[0:i], state[0:i, 0])
        anim_theta_sa.set_data(t[0:i], state_sa[0:i, 0])
        anim_omega.set_data(t[0:i], state[0:i, 1])
        anim_omega_sa.set_data(t[0:i], state_sa[0:i, 1])
        anim_pendulum.set_data([0, x_y[i, 0]], [0, x_y[i, 1]])
        anim_trajectory.set_data(x_y[0:i, 0], x_y[0:i, 1])
        anim_phase.set_data(state[0:i, 0], state[0:i, 1])

        if t[i] > t_window:
            ax[0, 0].set_xlim(t[i]-t_window, t[i])
            ax[0, 1].set_xlim(t[i]-t_window, t[i])

    # Call the animator
    anim_fargs = (state, state_sa, x_y, x_y_sa)
    anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                   init_func=init, fargs=anim_fargs,
                                   interval=10)
    return anim


anim = execute_part2()
plt.show()
