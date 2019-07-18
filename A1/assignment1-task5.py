import numpy as np
import matplotlib.pyplot as plt
plt.ioff()
from sklearn.metrics.pairwise import pairwise_distances as pairdist
import scipy.spatial.distance
import copy as cp
from matplotlib import rc
rc('text', usetex=True)

# Task 5: Gradient Descent algorithm
inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
outputs = np.array([0., 1., 1., 0.])

def sigmoid(x):
    #sigmoid function
    return 1./(1.+ np.exp(-x))
    
def tanh(x):
    #tangent function
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    
def relu(x):
    #linear rectifier
    if np.isscalar(x):
        if x < 0:
            x = 0
        return x
    else:
        return np.array([0 if i < 0 else i for i in x])      

def sign(x):
    #step function
    if x >.5:
        x = 1.
    if x <=.5:
        x = 0.
    return x

def xor_net(x, w, func):
    """
    Calculate output of xor-function
    Input:
        x - [x1, x2] input
        w - [w0, .., w8] weights
        func - activation function
    Returns:
        out - predicted output of xor function (scalar)
    """
    x = np.insert(x, 0, 1.)
    h = np.dot(w[:6].reshape(2,3), x)
    h = func(h)
    out = np.dot(w[6:], np.insert(h, 0, 1.))
    return func(out)

def mse(w, func):
    """
    Calculate mean squared error over all inputs: 
        mse = squared difference between predicted output and actual output of xor function for all inputs
    Input:
        w - [w0, ..., w8] weights
        func - activation function
    Returns:
        s - mean squared error over all outputs
    """
    s = 0.
    for i in range(len(outputs)):
        pd = (xor_net(inputs[i], w, func) - outputs[i])
        s += pd**2.
    return s / len(outputs)

def grdmse(w, func):
    """
    Calculate the gradient of the mean squared error for each weight
    Input:
        w - [w0, ..., w8] weights
        func - activation function
    Returns:
        gradient of mse
    """
    eps = 1e-3
    we = np.array([[j+eps if j==w[i] else j for j in w] for i in range(len(w))])
    return np.array([(mse(we[i], func) - mse(w, func))/eps for i in range(len(w))])

def Accuracy(fit, data_out):
    """
    Calculate the percentage of correctly classified inputs
    Input:
        fit - predicted output
        data_out - desired output
    Returns:
        percentage
    """
    return len([i for i, j in zip(fit, data_out) if i == j])/len(data_out)*100.

def update_weights(func, w, eta):
    """
    Apply perceptron algorithm to update weights of network
    Input:
        func - activation function
        w - [w0, ..., w8] weights
        eta - stepsize
    Calculates
        y - predicted output
        t - time
    Returns:
        mses - mean squared error for every run
        acc - accuracy for every run
    """    
    y = [sign(xor_net(inputs[i], w, func)) for i in range(len(inputs))]
    t = 0
    mses = [mse(w, func)]
    acc = [Accuracy(y, outputs)]
    while not (np.array_equal(outputs, y) or t == 5000):
        w -= eta * grdmse(w, func)
        y = [sign(xor_net(inputs[i], w, func)) for i in range(len(inputs))]
        mses.append(mse(w, func))
        acc.append(Accuracy(y, outputs))
        t += 1
        if t%10 == 0:
            print(t)
            print("Accuracy", Accuracy(y, outputs))
    print("Training completed: 100% accuracy reached")
    return mses, acc
 
# Apply Gradient Descent Algorithm with different activation functions, weights and stepsizes
np.random.seed(0) ; weights = np.random.rand(9) ; mse_arr_sigm, accuracy_sigm = update_weights(sigmoid, weights, eta=1.)   
np.random.seed(4) ; weights = np.random.randn(9) ; mse_arr_tanh, accuracy_tanh = update_weights(tanh, weights, eta=0.5)    
np.random.seed(2) ; weights = np.random.normal(loc=0., scale=0.2, size=9) ; mse_arr_relu, accuracy_relu = update_weights(relu, weights, eta=.005)

# Plot accuracy and mean squared error per run for different activation functions
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,3), tight_layout=True)
ax4 = ax1.twinx() ; ax5 = ax2.twinx() ; ax6 = ax3.twinx()
ax1.set_title('sigm $\eta = 1$') ; ax2.set_title('tanh $\eta = 0.5$') ; ax3.set_title('relu $\eta = 0.005$')
ax1.plot(np.arange(len(accuracy_sigm)), accuracy_sigm, color='orange', label='accuracy', linewidth=.5)
ax2.plot(np.arange(len(accuracy_tanh)), accuracy_tanh, color='orange', linewidth=.5)
ax3.plot(np.arange(len(accuracy_relu)), accuracy_relu, color='orange', linewidth=.5)
ax4.plot(np.arange(len(mse_arr_sigm)), mse_arr_sigm, color='blue', label='mse', linewidth=.5)
ax5.plot(np.arange(len(mse_arr_tanh)), mse_arr_tanh, color='blue', linewidth=.5)
ax6.plot(np.arange(len(mse_arr_relu)), mse_arr_relu, color='blue', linewidth=.5)
ax1.set_xlabel('run') ; ax2.set_xlabel('run') ; ax3.set_xlabel('run')
ax1.set_ylabel('Accuracy (\%)') ; ax6.set_ylabel('Mean squared error')
fig.legend()
fig.savefig("gradient-descent.png")
plt.show()
