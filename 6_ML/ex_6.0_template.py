"""
We intend to use linear regression to fit a proxy dataset of linearly
correlated data, as a proof of concept of the descriptive and predictive power
of regression.
We will generate a set of Ndata linearly correlated data, i.e. a collection of
Ndata (x,y) points such that y = a * x + b except for noise, and we'll use a
gradient descent algorithm to fit the data with a linear function
y_pred(x) = a_pred * x + b_pred.

We'll then replace the gradient descent algorithm with a simple neural network,
we will train it on the generated dataset and we'll compare the results with
the ones obtained with a gradient descent.

We'll then introduce multiple hidden nodes and a non-linear activation function
in the neural network's architecture to perform non-linear regressions,
and we'll fit a proxy gaussian dataset.
"""
from typing import List, Tuple
import sys
import numpy as np
import matplotlib.pyplot as plt


# Method that generates a linear dataset
def generate_linear_data(
        ndata: int, interval: Tuple[int, int],
        a: float, b: float, delta: float
    ) -> np.array:
    """
        ndata: number of data to generate
        interval: tuple (min_x, max_x) setting the interval of values of x
        a: slope of the linear dependence
        b: intercept of the linear dependence
        delta: amplitude of the noise

        Routine that generates a set of linearly correlated ndata (x, y) points,
        generating points that verify the equation y = a * x + b and
        introducing noise.

        Returns the set of data in 2d numpy array of shape 2 x ndata.
    """
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
    # Generate x values uniformly distributed in the interval;
    # x is a 1d numpy array of shape (ndata, )
    # x = np.random.uniform(interval[0], interval[1], ndata)
    x = np.linspace(interval[0], interval[1], ndata)

    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
    # Generate y values with normally distributed noise;
    # y is a 1d numpy array of shape (ndata, )
    y = a * x + b + np.random.normal(0, delta, ndata)

    # https://numpy.org/doc/stable/reference/generated/numpy.vstack.html
    # Stack x and y sequences row-wise
    return np.vstack((x, y))


def execute_part1():
    # Set the number of data and the interval for the x axis
    global ndata
    ndata =  11         # 10 + 1 data points
    global interval
    interval = (0, 10)

    # Set the slope of the linear correlation
    global a
    a = 2

    # Set the amplitude of the noise
    global delta
    delta = 5

    # Generate the data
    global data
    data = generate_linear_data(ndata, interval, a, 0, delta)
    # print(type(data))

    # Then, plot the data
    plt.plot(data[0], data[1], 'o', label=r'$(x_i,y_i)$')
    plt.title('Linear dataset')
    plt.xlabel('x')
    plt.ylabel('y')
    xs = np.linspace(interval[0], interval[1], 100)
    plt.plot(xs, xs * a, '-', label=r'$y = {}x$'.format(a))
    plt.legend()
    plt.show()


# Method that calculates the cost-function J (average squared residuals)
def J(data: np.array, a: float, b: float) -> float:
    """
        data: dataset
        a, b: intercept and slope of the linear model

        Routine that calculates the average squared residuals with respect
        of the dataset against the linear model y = a * x + b.
    """ # As per Section 8 of the reader
    linear_model = a * data[0] + b     # a line with slope a and intercept b
    residuals = linear_model - data[1] # error between the model and the data
    J = 0.5 * np.mean(residuals**2)    # average squared residuals (with the
                                       # 0.5 kept, because of convention reasons)
    return J


# Method that calculates the derivatives of J
def J_derivatives(data: np.array, a: float, b: float) -> Tuple[float, float]:
    """
        data: dataset
        a, b: intercept and slope of the linear model

        Routine that calculates the derivatives of the cost function J
        with respect to a and b, and returns them as a tuple.
    """ # As per Section 8 of the reader
    linear_model = a * data[0] + b     # a line with slope a and intercept b
    residuals = linear_model - data[1] # error between the model and the data
                                       # average of residuals times x-values
    dJda = np.mean(residuals * data[0])
    dJdb = np.mean(residuals)          # average of residuals
    return (dJda, dJdb)                # return Tuple(derivatives)


# Method that performs one step of the gradient descent
def gradient_descent_step(
        data: np.array, a: float, b: float, alpha: float
    ) -> Tuple[float, float]:
    """
        data: dataset
        a, b: intercept and slope of the current best-fit
        alpha: learning rate

        Routine that performs one step of the gradient descent with
        learning parameter alpha, and returns the updated values
        of a and b as a tuple.
    """
                                        # calculate the derivatives
                                        # of the cost function
    dJda, dJdb = J_derivatives(data, a, b)
                                        # do gradient descent step
    a_new = a - alpha * (dJda)
    b_new = b - alpha * (dJdb)

    return (a_new, b_new)


def execute_part2():
    # Generate two random starting values for a and b
    global a_pred
    global b_pred
    a_pred = -1     # start with negative slope
    b_pred = 2      # set b as 2

    # Set a value for the learning rate
    alpha =  0.1    # start with 0.1 as learning rate

    # Then, perform gradient descent steps until convergence
    convergence_treshold = 0.000001
    errors = [J(data, a_pred, b_pred)]
    while True:
        a_pred, b_pred = gradient_descent_step(
            data, a_pred, b_pred, alpha
        )
        errors.append(J(data, a_pred, b_pred))

        if(np.abs(errors[-1] - errors[-2]) < convergence_treshold):
            break

    # Plot the errors along the evolution
    plt.figure()
    plt.title('Gradient descent')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.plot(range(len(errors)), errors)
    plt.show()

    # Then, plot the dataset with the fitted model
    plt.figure()
    plt.title('Gradient descent')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(data[0], data[1], 'o', label=r'$(x_i,y_i)$')
    xs = np.linspace(interval[0], interval[1], ndata)
    plt.plot(xs, xs * a_pred + b_pred, '-', label=r'Linear fit')
    plt.legend()
    plt.show()

    # And print the predicted values of a and b and their error with respect
    # to the real values
    print("Model: y = {} * x".format(a))
    print("Fit: y = {} * x + {}".format(a_pred, b_pred))
    print("Errors: a_err = {}, b_err = {}".format(np.abs(a_pred - a),
                                                  np.abs(b_pred)))


# Forward method for the simple neural network of fig 1.3
def NN1_forward(x: float, w: float, b: float) -> float:
    """
        x: input
        w: weight of the hidden node
        b: bias of the hidden node

        Routine that returns the output of the simple one-node NN of fig 1.3.
    """
    return w * x + b
    

# Method that performs a training step of the simple neural network of fig 1.3
def NN1_train(
        inputs: np.array, targets: np.array, w: float, b: float, alpha: float
    ) -> Tuple[float, float]:
    """
        inputs: inputs to the NN (just x-values)
        targets: targets of the NN
        w: weight of the hidden node
        b: bias of the hidden node
        alpha: learning rate

        Function that backpropagates the errors on the hidden node and returns
        the updated values of his weight and bias as a tuple
    """
    #! Wat moet ik met de variables parameter doen? 
    predictions = NN1_forward(inputs, w, b) # get the predictions
    data = np.array([inputs, predictions])
                                            # calculate derivatives (just like
    dJda, dJdb = J_derivatives(data, w, b)  # in the gradient descent function)
    w_new = w - alpha * (dJda)              # do gradient descent step
    b_new = b - alpha * (dJdb)

    return (w_new, b_new)


def execute_part3():
    # Generate a random starting values for w1
    global w
    w = 1       # start with 1

    # Set a value for the learning rate
    learning_rate = 0.1 # start with 0.1

    # Then, train the NN until convergence
    convergence_treshold = 0.000001
    errors = [J(data, w, 0)]
    ws = [w]
    while True:
        # For each training step, update your weights;
        # b is kept stable at 0, to investigate the behaviour of w
        w, _ = NN1_train(data[0], data[1], w, 0, learning_rate)

        errors.append(J(data, w, 0))
        ws.append(w)

        if(np.abs(errors[-1] - errors[-2]) < convergence_treshold):
            break

    # Plot the errors and w1 along the training
    plt.figure()
    plt.title('Training NN1')
    plt.xlabel('Epoch')
    plt.plot(range(len(errors)), errors, label='Error')
    plt.plot(range(len(errors)), ws, label='w1')
    plt.legend()
    plt.show()

    # Plot the errors as a function of w1
    plt.figure()
    plt.title('Training NN1')
    plt.xlabel('w1')
    plt.ylabel('Error')
    plt.plot(ws, errors)
    plt.show()

    # Then, plot the dataset with the fitted model
    plt.figure()
    plt.title('NN1')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(data[0], data[1], 'o', label=r'$(x_i,y_i)$')
    xs = np.linspace(interval[0], interval[1], 100)
    plt.plot(xs, NN1_forward(xs, w, 0), '-', label=r'NN')
    plt.legend()
    plt.show()


# # Method that generates a linear dataset
# def generate_gaussian_data(ndata, interval, x0, sigma2, delta):
#     """
#         ndata: number of data to generate
#         interval: tuple (min_x, max_x) setting the interval of values of x
#         x0: centre of the gaussian distribution
#         sigma2: standard deviation of the distribution
#         delta: amplitude of the noise

#         Routine that generates a set of ndata (x,y) points that verify the
#         equation
#         y = exp(-(x - x0)^2 / 2 * sigma2) and introducing noise.

#         Returns the set of data in 2d numpy array of shape ndata x 2.
#     """

#     # YOUR CODE GOES HERE

#     return np.vstack((x, y))

# # Activation function


# def g(z):
#     # YOUR CODE GOES HERE

#     # Derivative of the activation function


# def dg(z):
#     # YOUR CODE GOES HERE

#     # Forward method for the neural network of fig 1.5


# def NN2_forward(x, w1, b1, w2):
#     """
#         x: input (a single scalar)
#         w1: array of weights of the hidden layer
#         b1: array of biases of the hidden layer
#         w2: array of weights of the output layer

#         Routine that returns the output of the NN of fig 1.5.
#         The output should be a single scalar, just like the input x
#     """

#     # YOUR CODE GOES HERE

# # Forward method for the neural network of fig 1.5


# def NN2_error(inputs, targets, w1, b1, w2):
#     """
#         x: input
#         w1: array of weights of the hidden layer
#         b1: array of biases of the hidden layer
#         w2: array of weights of the output layer

#         Routine that returns the error of the NN of fig 1.5.
#     """

#     # YOUR CODE GOES HERE

# # Method that performs a training step of the neural network of fig 1.5


# def NN2_train(inputs, targets, w1, b1, w2, dw1, db1, dw2, lr, momentum, wd):
#     """
#         inputs: inputs to the NN
#         targets: targets of the NN
#         w1: array of weights of the hidden layer
#         b1: array of biases of the hidden layer
#         w2: array of weights of the output layer
#         dw1: array of last updates of the hidden layer weights
#         db1: array of last updates of the hidden layer biases
#         dw2: array of last updates of the output layer weights
#         lr: learning rate
#         momentum: momentum
#         wd: weight decay

#         Function that backpropagates the errors on the hidden nodes and returns
#         the updated values of their weights and biases as a tuple of arrays
#     """

#     # YOUR CODE GOES HERE

#     return (w1_new, b1_new, w2_new, dw1_new, db1_new, dw2_new)


# def execute_part4():
#     # Set the number of data and the interval for the x axis
#     global ngdata
#     ngdata = 500
#     global ginterval
#     ginterval = (-2.5, 2.5)

#     # Set the amplitude of the noise
#     global gdelta
#     gdelta = 0.1

#     # Generate the data
#     global gaussian_data
#     gaussian_data = generate_gaussian_data(ngdata, ginterval, 0, 0.5, gdelta)

#     # Then, plot the data
#     plt.figure()
#     plt.plot(gaussian_data[0], gaussian_data[1], 'o', label=r'$(x_i,y_i)$')
#     plt.title('Gaussian dataset')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     xs = np.linspace(ginterval[0], ginterval[1], 100)
#     plt.plot(xs, np.exp(-(xs)**2), '-', label=r'$y = e^{-x^2}$')
#     plt.legend()
#     plt.show()

#     # Now, for different numbers of hidden layers
#     final_errors = []
#     final_fits = {}
#     ns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     for n in ns:
#         nhidden = n
#         # And generate starting values for the hidden layer weights and biases
#         # with the Xavier initialization
#         global w1, b1, w2
#         w1 =  # YOUR CODE GOES HERE
#         b1 =  # YOUR CODE GOES HERE
#         w2 =  # YOUR CODE GOES HERE
#         dw1 =  # YOUR CODE GOES HERE
#         db1 =  # YOUR CODE GOES HERE
#         dw2 =  # YOUR CODE GOES HERE

#         # Set a value for the learning rate, momentum and weight decay
#         learningrate = 0.1
#         momentum = 0.9
#         weight_decay = 0.0001

#         # Then, train the NN for nepochs epochs (epoch=a training step)
#         nepochs = 1000
#         errors = [NN2_error(gaussian_data[0], gaussian_data[1], w1, b1, w2)]
#         for epoch in range(nepochs):
#             # For each training step, update your weights

#             # Update the list of errors
#             errors.append(NN2_error(gaussian_data[0], gaussian_data[1], w1, b1, w2))

#         # Plot the errors and w1 along the training
#         plt.figure()
#         plt.title('Training with {} hidden layers'.format(nhidden))
#         plt.xlabel('Epoch')
#         plt.ylabel('Error')
#         plt.plot(range(len(errors)), errors)
#         final_errors.append(errors[-1])
#         plt.show()

#         # Then, plot the dataset with the fitted model
#         plt.figure()
#         plt.title('Curve fit with {} hidden layers'.format(nhidden))
#         plt.xlabel('x')
#         plt.ylabel('y')
#         plt.plot(gaussian_data[0], gaussian_data[1], 'o', label=r'(x_i,y_i)')
#         xs = np.linspace(ginterval[0], ginterval[1], 100)
#         y_pred = []
#         for x in xs:
#             y_pred.append(NN2_forward(x, w1, b1, w2))
#         final_fits[n] = np.array(y_pred)
#         plt.plot(xs, np.array(y_pred), '-', label=r'NN')
#         plt.show()

#     # Plot the errors trend
#     plt.figure()
#     plt.xlabel('N hidden nodes')
#     plt.ylabel('Error')
#     plt.plot(ns, final_errors, '-o')
#     plt.show()
#     # And the fits
#     plt.figure()
#     plt.xlabel('Curve fits')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.plot(gaussian_data[0], gaussian_data[1], 'o', label=r'(x_i,y_i)')
#     xs = np.linspace(ginterval[0], ginterval[1], 100)
#     for n in ns:
#         plt.plot(xs, final_fits[n], '-', label=r'n = {}'.format(n))
#     plt.legend()
#     plt.show()


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
    # if validate_args(sys.argv):
    #     part = int(sys.argv[1])
    execute_part1()
    execute_part2()
    execute_part3()
   


if __name__ == "__main__":
    main()