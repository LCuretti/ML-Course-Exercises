# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:26:35 2019

@author: LCuretti
"""

import numpy as np
from numpy import loadtxt

import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


# WARMUPEXERCISE Example function in octave
#   A = WARMUPEXERCISE() is an example function that returns the 5x5 identity matrix

A = []

# ============= YOUR CODE HERE ==============
# Instructions: Return the 5x5 identity matrix 
#               In octave, we return values by defining which variables
#               represent the return values (at the top of the file)
#               and then set them accordingly. 

A = np.identity(5)

# ===========================================

A


# Loading data file
data = loadtxt("ex1data1.txt", comments="#", delimiter=",", unpack=False)

X = data[:, 0]
y = data[:, 1]
m = len(X) # number of training examples


def plotData(x, y):
#PLOTDATA Plots the data points x and y into a new figure 
#   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
#   population and profit.

#figure; % open a new figure window

# ====================== YOUR CODE HERE ======================
# Instructions: Plot the training data into a figure using the 
#               "figure" and "plot" commands. Set the axes labels using
#               the "xlabel" and "ylabel" commands. Assume the 
#               population and revenue data have been passed in
#               as the x and y arguments of this function.
#
# Hint: You can use the 'rx' option with plot to have the markers
#       appear as red crosses. Furthermore, you can make the
#       markers larger by using plot(..., 'rx', 'MarkerSize', 10);

    ax1 = plt.subplot(1,1,1)
    ax1.plot(x, y, 'r+')
     
    plt.ylabel('Profit in $10,000s') # Set the y−axis label
    plt.xlabel('Population of City in 10,000s') # Set the x−axis label
    plt.show()

# ============================================================

    return ax1

plt1 = plotData(X,y)


X = X.reshape(m,1) #changing from (97,) to (97,1) 
y = y.reshape(m,1)
X = np.hstack((np.ones((m, 1)),X)) # Add a column of ones to x

theta = np.zeros((2, 1)) # initialize fitting parameters
iterations = 1500
alpha = 0.01


def computeCost(X, y, theta):
# =============================================================================
# COMPUTECOST Compute cost for linear regression
#    J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
#    parameter for linear regression to fit the data points in X and y
# 
#  Initialize some useful values
# =============================================================================
    m = len(y) # number of training examples

# You need to return the following variables correctly 
    J = 0

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.

    J = sum(((np.dot(X, theta).reshape(m,1))-y)**2)/m*.5

# =========================================================================

    return J

J = computeCost(X, y, theta)


def gradientDescent(X, y, theta, alpha, num_iters):
    
# GRADIENTDESCENT Performs gradient descent to learn theta
#   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
#   taking num_iters gradient steps with learning rate alpha

# Initialize some useful values
    
    m = len(y) # number of training examples
    J_history = np.zeros((num_iters, 1))

    for iter in range(0,num_iters):

    # ====================== YOUR CODE HERE ======================
    # Instructions: Perform a single gradient step on the parameter vector
    #               theta. 
    #
    # Hint: While debugging, it can be useful to print out the values
    #       of the cost function (computeCost) and gradient here.
    #

        He = (np.dot(X, theta).reshape(m,1))-y
        #R1 = alpha/m*np.dot(He.T,X[:,0])
        #R2 = alpha/m*np.dot(He.T,X[:,1])
        
        #R1 = alpha/m*np.dot(((np.dot(X, theta).reshape(m,1))-y).T,X[:,0])
        #R2 = alpha/m*np.dot(((np.dot(X, theta).reshape(m,1))-y).T,X[:,1])

        #theta[0,0]= theta[0,0] - R1
        #theta[1,0]= theta[1,0] - R2

        theta = np.array(((theta[0,0] - alpha/m*np.dot(He.T,X[:,0])), (theta[1,0] - alpha/m*np.dot(He.T,X[:,1]))))
        
    # ============================================================

    # Save the cost J in every iteration    
        J_history[iter] = computeCost(X, y, theta)

    return theta, J_history

theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
print ('Theta computed from gradient descent: {}, {}'.format(theta[0],theta[1]))


plt1.plot(X[:,1],np.dot(X,theta), '-')

plt1.legend(('Training data','Linear regression'))


predict1 = np.dot([1, 3.5],theta)
print('For population = 35,000, we predict a profit of', predict1[0]*10000)

predict2 = np.dot([1, 7],theta)
print('For population = 70,000, we predict a profit of', predict2[0]*10000)


# Visualizing J(theta_0, theta_1):
# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)


# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals
for i in range (0,100):
    for j in range (0,100):
        t = np.array((theta0_vals[i],theta1_vals[j]))
        J_vals[i,j] = computeCost(X, y, t)[0]

# =============================================================================
# # Surface plot, need to create the grid to get the proper surface 

Xg,Yg = np.meshgrid(theta1_vals,theta0_vals)
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.set_xlim(4, -1)
ax.set_xlabel('Theta 1') 
ax.set_ylim(-10, 10)
ax.set_ylabel('Theta 0')
ax.set_zlim(0, 800)
ax.plot_surface(Xg, Yg, J_vals, cmap=cm.coolwarm)

# # Contour plot
# # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
 
Z = np.log(J_vals)

fig, ax = plt.subplots()
cs = ax.contour(Yg, Xg, Z)
ax.set_xlabel('Theta 0') 
ax.set_ylabel('Theta 1')
cbar = fig.colorbar(cs)
plt.plot(theta[0], theta[1], 'rx')

