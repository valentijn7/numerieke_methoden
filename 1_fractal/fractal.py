# 1_fractal/fractal.py

import sys
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#! Vraag hoe het zit met de complexe getallen in de opdracht, en ook
#! hoe het zit met afronden enzo
# Manual: to run the various parts of the code, use the following command:
# python fractal.py <opdracht>
# where <opdracht> is an integer between 1 and 5, corresponding to the exercise

############# PART 1 ##############
"""
The Newton's fractal can be determined using the Newton's method of root finding.
In this exercise we will obtain the fractal by solving nemrically polynomials with complex roots.
We will start with simple polynomials.

Using the bisection, secant or Brent's method
determine numerically the roots for
f(x) = (+1-x) * (-1-x)
"""

def f(x: float, roots: List[float]) -> float:
    """
    f: polynomial
    x: evaluate f at x
    roots: numpy array of roots of the polynomial

    Returns the value of f(x).
    """
    # np.prod returns product of (x minus root) for each root in the array
    return np.prod([x - root for root in roots])
    

def bisection_method(roots: List[float], interval: Tuple[float, float], verbose: bool = False):
    """
    roots: list of polynomial roots
    interval: a tuple (a,b) defining the starting interval which brakets the solution
    
    Function that determines the root of the equation f(x) = 0 within the interval (a, b) using the method
    of bisection.

    Returns a tuple (Root, Error, Number of Iterations)
    """
    if verbose:
        print('bisection_method() with roots:', roots, 'interval:', interval)

    a, b = interval[0], interval[1] # unpack tuple to get a and b
    dx = abs(b - a)                 # determine dx (= b - a)
    
    while dx > 1.0e-6:              # set tolerance to 0.000001
        x = (a + b) / 2             # bisect to get x
                                    # if f(a) and f(x) have different signs, set b = x
        if f(a, roots) * f(x, roots) < 0:
            b = x
        else:                       # else, set a = x
            a = x
        dx = abs(b - a)             # update dx
        
    if verbose:
        print(f'Found f(x) = 0 at x = {x:2f} with error {dx:2f}')
        print()
        
    
# # Build the method 
# def secant_method(roots, interval):
#     """
#     roots: list of polynomial roots
#     interval: a tuple (a,b) defining the starting interval, which in this case does not have to include the solution
#               but just to be close to it
    
#     Function that determines the root of the equation f(x) = 0 within the interval (a, b) using the secant method.

#     Returns a tuple (Root, Error, Number of Iterations)
#     """
#     For I, I used the bisection method
    

# # Build the method 
# def brent_method(roots, x0):
#     """
#     roots: list of polynomial roots
#     x0: an array [x0, x1, x2] defining the initial guesses
    
#     Function that determines the root of the equation f(x) = 0 within the interval (a, b) using the Brent's method.

#     Returns a tuple (Root, Error, Number of Iterations)
#     """
#     # For I, I used the bisection method


def execute_part_1():
    """
    Executes ex. I.1; passes the roots and intervals to the bisection method,
    which, in turn, prints out the information (which is copied below the
    function definition for convenience).
    """
    roots = [1, -1]
    intervals = [(0.5, 2), (-2, -0.5), (-2, 2), (1.5, 3)]
    
    for interval in intervals:
        bisection_method(roots, interval, verbose = True)
    
    ### Outputs:
    """
    bisection_method() with roots: [1, -1] interval: (0.5, 2)
    Found f(x) = 0 at x = 1.000000 with error 0.000001

    bisection_method() with roots: [1, -1] interval: (-2, -0.5)
    Found f(x) = 0 at x = -1.000000 with error 0.000001

    bisection_method() with roots: [1, -1] interval: (-2, 2)
    Found f(x) = 0 at x = -0.000001 with error 0.000001

    bisection_method() with roots: [1, -1] interval: (1.5, 3)
    Found f(x) = 0 at x = 2.999999 with error 0.000001
    """


############# PART 3 ##############
"""
The exact same polynomial can be solved using Newton's method.
Notice that in this case we also need to determine the derivative of the polynomial.

For initial guess use the folowing values
x0 = +2, +10, 0.5, 0

To what root does each guess converge?
"""


def df(x: float, roots: List[float], dx: float = 1e-6) -> float:
    """
    We use numerical differentiation to calculate the derivative at x
    of a polynomial f(x) defined

    f(x) = (r1 - x) * (r2 - x) * ... * (rn - x)

    where r1, r2, ..., rn are the roots of the polynomial (see Eq. (59)
    in the lecture notes). To obtain f'(x), we differentiate with
    the 'symmetric three-point formula', Eq. (13) in the lecture notes.
    (Only the first term on the right-hand side of Eq. (13) is used here.)

    :param x : the point at which to evaluate the derivative
    :param roots : the roots of the polynomial
    :param dx : the step size for differentiation
    :return : the (polynomial) derivative of f at x
    """
    return (f(x + dx, roots) - f(x - dx, roots)) / (2 * dx)


def execute_part_2():
    """
    Executes ex. 1.2, by checking a few deritatives at different points for
    a general polynomial (as defined in Eq. (59) in the lecture notes).
    """
    roots = [1, -1]
    x_values = [0, 1, 2, 3]
    for x in x_values:
        print(f"f'({x}) = approx. {df(x, roots)}")

    ### Outputs:
    """
    f'(0) = approx. 0.0
    f'(1) = approx. 1.999999999946489
    f'(2) = approx. 4.000000000337067
    f'(3) = approx. 6.000000000838668
    """


def round_guess(x: float, error: float, n: int, dec: int = 6) -> Tuple[float, float, int]:
    """
    Helper function to newton_raphson_method(); it rounds the result to
    the desired precision, adjust approximated but unprecise imaginary
    parts, and returns the result as a tuple (Root, Error, Number of Iterations)

    :param x : the root found by the Newton-Raphson method
    :param error : the error in the root
    :param n : the number of iterations
    :param dec : the number of decimals to round the result to
    """
    # force imaginary part to be zero if it is very small
    if isinstance(x, complex) and abs(x.imag) < 1.0e-6:
        x = x.real
    return np.round(x, dec), np.round(error, dec), n



def newton_raphson_method(
        roots: List[float], x0: float, maxiter: int = 1000, dec: float = 6
    ) -> Tuple[float, float, int]:
    """
    Function that determines the root of the equation f(x) = 0 within
    the interval (a, b) using the Newton-Raphson method.

    :param roots : list of polynomial roots
    :param x0 : double with the initial guess
    :param maxiter : maximum iterations (to prevent infinite loops)
    :param dec : number of decimals to round the result to
    :return : a tuple (Root, Error, Number of Iterations)
    """
    n = 0
    tolerance = 1.0e-6

    while n < maxiter:
        fx = f(x0, roots)       # calculate f(x0)
        dfx = df(x0, roots)     # calculate f'(x0)
        if abs(dfx) < 1.0e8:    # if f'(x0) is smaller than 1.0e8, perturb it slightly
                                # into the complex plane to prevent division by zero
            dfx += (1.0e-6 + 1.0e-6j)
        x1 = x0 - fx / dfx      # calculate x1 (as per Eq. (2))
                                # if within the error tolerance, return result
        if abs(x1 - x0) < tolerance:
            return round_guess(x1, abs(x1 - x0), n, dec)
        
        x0 = x1                 # update x0 and n
        n += 1
                                # if the loop exits, the method did not converge
    # print(f'newton_raphson_method(): did not converge after {n} iterations')
    return x1, abs(x1 - x0), n
#! For the report, keep in mind the story about the Real and Imaginary parts, and how
#! to choose the thresholds/precision etc., besides all the normal stuff to tell

def execute_part_3():
    """
    Try to find a solution using the Newton's method for Eq. (60),
    which equivalently reads: f(x) = (x + 1) * (x - 1). To which
    root does each initial guess converge?
    """
    roots = [-1, 1] 
    initial_guesses = [-2, 10, 0.5, 0]

    print('\nRoot, Error, Iterations')
    for x0 in initial_guesses:
        print(newton_raphson_method(roots, x0))
    print()

    ### Outputs:
    """
    Root, Error, Iterations
    (-1.0, 0.0, 4)
    (1.0, 0.0, 7)
    (1.0, 0.0, 4)
    (1.0, 0.0, 24)
    """


############# PART 4 ##############
"""
The function you developed before should be valid also for complex numbers.

Try to solve the complex polynomial
f(z) = (1 + 1j - z) * (1 - 1j - z)
Notice that 1j corresponds to the complex number i

Use the following initial guesses
z0 = 2+1j, -1j, 3, 2
To what root each guess converges? Does it always return a correct root?
"""

def execute_part_4():
    """
    Executes ex. 1.4, by checking the convergence of the Newton's method
    for a complex polynomial, using the initial guesses from above
    """
    roots = [1 + 1j, 1 - 1j]
    initial_guesses = [2 + 1j, -1j, 3, 2]

    print('\nRoot, Error, Iterations')
    for x0 in initial_guesses:
        print(newton_raphson_method(roots, x0))
    print()

    ### Outputs:
    """
    Root, Error, Iterations
    ((1+1j), 0.0, 5)
    ((1-1j), 0.0, 5)
    ((1+1j), 1e-06, 24)
    ((1+1j), 0.0, 24)
    """

############# PART 5 ##############
"""
To draw the Newton's fractal we need to make a numpy array with the root
that converged from several initial guesses.

Consider the polynomial
f(z) = (1 + 1j - z) * (+1 - 1j - z) * (-1 + 1j - z) * (-1 - 1j - z)

On the numpy array Z save an integer that corresponds
to the closest root to the value returned by the Newton's method.
For example
Z = [[2, 1],
     [0, 3]]
is the numpy array that Newton's method returned a value that is closest to the roots
[[-1 + 1j, +1 - 1j],
 [+1 + 1j, -1 - 1j]]

Notice: to save computing time we will only consider maximum 4 Newton iterations.
"""

def execute_part_5():
    """
    Executes ex. 1.5, by plotting the Newton's fractal for the polynomial
    f(z) = (1 + 1j - z) * (+1 - 1j - z) * (-1 + 1j - z) * (-1 - 1j - z),
    using gerenal initial guesses
    """
    roots = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]

    x_min, x_max, nx = -2, +2, 10 # minimum, maximum value of x, and number of x points
    y_min, y_max, ny = -2, +2, 10 # minimum, maximum value of y, and number of y points

    maxiter = 4
    Z = np.empty((nx, ny))

    for idx, x in enumerate(np.linspace(x_min, x_max, nx)):
        for jdx, y in enumerate(np.linspace(x_min, x_max, nx)):
            # get initial guess and calculate the root with ... iterations
            x0 = x + y * 1j
            root, _, _ = newton_raphson_method(roots, x0, maxiter)

            # find the closest root, or, in other words, the root with
            # the smallest absolute difference to the calculated root
            closest_root = np.argmin([abs(root - r) for r in roots])
            Z[jdx, idx] = closest_root

    # plot script
    fig, ax1 = plt.subplots()

    plot = ax1.imshow(Z,
                      extent = (x_min, x_max, y_min, y_max),
                      vmin = 0,
                      vmax = len(roots) - 1,
                      interpolation = 'none')
    fig.colorbar(plot,
                 ax = ax1,
                 ticks = np.arange(len(roots)))
    ax1.scatter([r.real for r in roots],
                [r.imag for r in roots],
                color = 'k',
                marker = 'X',
                edgecolors = 'w')
    plt.savefig(f'newton_fractal_{maxiter}_iterations.pdf')
    plt.show()
    

def validate_args(args: List[str]) -> bool:
    """
    Does a quick check on whether the argument (a number between 1 and 5)
    is correctly and entered and thus valid. Else, it raises an error.
    
    :param args : a list of command line arguments
    :return : True if the argument is valid, False otherwise
    """
    if len(sys.argv) != 2 or not sys.argv[1].isdigit() or int(sys.argv[1]) not in range(1, 6):
        raise ValueError("Usage: python fractal.py <opdracht>"\
                         "\n<opdracht> should be an integer between 1 and 5")
    return True
    

def main():
    if validate_args(sys.argv):
        part = int(sys.argv[1])
    
    if part == 1:
        execute_part_1()
    elif part == 2:
        execute_part_2()
    elif part == 3:
        execute_part_3()
    elif part == 4:
        execute_part_4()
    elif part == 5:
        execute_part_5()
    

if __name__ == '__main__':
    main()