from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

def symmetric_five_point_formula(x: float, delta_x: float, f: ) -> float:
    """
    Calculates (15) in the lecture notes, the symmetric five-point formula

    :param x: the point at which to evaluate the derivative
    :param delta_x: the step size
    :param f: the function to differentiate
    :return: the derivative of f at x
    """

def df(x: float, roots: List[float]) -> float:
    """
    df: polynomial derivative
    x: evaluate f' at x
    roots: numpy array of roots of the polynomial

    We use numerical differentiation to calculate the derivative at x
    of a polynomial f(x) defined

    f(x) = (r1 - x) * (r2 - x) * ... * (rn - x)

    where r1, r2, ..., rn are the roots of the polynomial (see (59)
    in the lecture notes). To obtain f'(x), we differentiate with
    the 'symmetric five-point formula', (15) in the lecture notes.

    Returns the value of f'(x).
    """

    




def newton_method(roots, x0, maxiter=1000):
    """
    roots: list of polynomial roots
    x0: double with the initial guess
    maxiter: maximum iterations to prevent infinite loops
    
    Function that determines the root of the equation f(x) = 0 within the interval (a, b) using the Brent's method.

    Returns a tuple (Root, Error, Number of Iterations)
    """
    #YOUR CODE GOES HERE


def execute_part_3():
    #Try to find a solution using the Newton's method
    #YOUR CODE GOES HERE

    roots = [...] # array with list of roots

    x0 = 1
    newton_method(roots, x0)


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
    #Try to find a solution using the Newton's method
    #YOUR CODE GOES HERE

    roots = [...] # array with list of complex roots

    #Test your Newton's code with complex numbers
    x0 = 1+1j
    newton_method(roots, x0)

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
    #Try to obtain the Newton's fractal
    #YOUR CODE GOES HERE

    roots = [...] # array with list of complex roots

    x_min, x_max, nx = -2, +2, 10 # minimum, maximum value of x, and number of x points
    y_min, y_max, ny = -2, +2, 10 # minimum, maximum value of y, and number of y points

    Z = np.empty((nx, ny))

    for i, x in enumerate(np.linspace(x_min, x_max, nx)):
        for j, y in enumerate(np.linspace(x_min, x_max, nx)):
            root = newton_method(roots, x0, 4)

            # Z[j, i] = #YOUR CODE TO FIND THE CLOSEST ROOT GOES HERE


    # plot script
    fig, ax1 = plt.subplots()

    plot = ax1.imshow(Z, extent=(x_min, x_max, y_min, y_max), vmin=0, vmax=len(roots)-1, interpolation='none')
    fig.colorbar(plot, ax=ax1, ticks=np.arange(len(roots)))
    ax1.scatter(roots.real, roots.imag, color='k', marker='X', edgecolors='w')

    plt.show()
    

def main():
    execute_part_1()
    

if __name__ == '__main__':
    main()