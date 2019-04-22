# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 12:10:21 2019

@author: LCuretti
"""
import numpy as np
from numpy import loadtxt

import matplotlib.pyplot as plt

'''
###########################################################################
###############  Optional Exercises: ######################################
###########################################################################
'''

data = loadtxt("ex1data2.txt", comments="#", delimiter=",", unpack=False)
X = data[:, 0:2]
y = data[:, 2]
m = len(X) # number of training examples
y = data[:, 2].reshape(m,1)


# Print out some data points
# First 10 examples from the dataset
print(' x = {}\n, y = {}\n'.format(X[0:9,:], y[0:9,:]))


def featureNormalize(X):
    #FEATURENORMALIZE Normalizes the features in X 
    #   FEATURENORMALIZE(X) returns a normalized version of X where
    #   the mean value of each feature is 0 and the standard deviation
    #   is 1. This is often a good preprocessing step to do when
    #   working with learning algorithms.
    
    # You need to set these values correctly
    
    X_norm = np.copy(X)
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: First, for each feature dimension, compute the mean
    #               of the feature and subtract it from the dataset,
    #               storing the mean value in mu. Next, compute the 
    #               standard deviation of each feature and divide
    #               each feature by it's standard deviation, storing
    #               the standard deviation in sigma. 
    #
    #               Note that X is a matrix where each column is a 
    #               feature and each row is an example. You need 
    #               to perform the normalization separately for 
    #               each feature. 
    #
    # Hint: You might find the 'mean' and 'std' functions useful.
    #       
    
    
    for i in range (0,X.shape[1]):
        mu[0,i] = np.mean(X_norm[:,i])
        X_norm[:,i] = X_norm[:,i]-mu[0,i]
        sigma[0,i] = np.std(X_norm[:,i])
        X_norm[:,i] = X_norm[:,i]/sigma[0,i]
  
    # ============================================================
    
    return X_norm, mu, sigma

X, mu, sigma = featureNormalize(X)


# Add intercept term to X
X = np.hstack((np.ones((m, 1)),X))


def computeCostMulti(X, y, theta):
    #COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
    #   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
    #   parameter for linear regression to fit the data points in X and y
    
    # Initialize some useful values
    m = len(y); # number of training examples
    
    # You need to return the following variables correctly 
    J = 0;
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.
    
    J = sum(((np.dot(X, theta).reshape(m,1))-y)**2)/m*.5
    
    
    
    # ==================================================================

    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    #GRADIENTDESCENTMULTI Performs gradient descent to learn theta
    #   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
    #   taking num_iters gradient steps with learning rate alpha
    
    # Initialize some useful values
    m = len(y); # number of training examples
    J_history = np.zeros((num_iters, 1))
    
    for iter in range (0,num_iters):
    
        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta. 
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCostMulti) and gradient here.
        #
        
        He = (np.dot(X, theta).reshape(m,1))-y
        
        for i in range (0,theta.shape[0]):
            theta[i,0]= theta[i,0] - alpha/m*np.dot(He.T,X[:,i])
      
        # ============================================================
    
        # Save the cost J in every iteration    
        J_history[iter] = computeCostMulti(X, y, theta)

    return theta, J_history

alpha = 0.1
num_iters = 400

# Init Theta and Run Gradient Descent 
theta = np.zeros((3, 1))
theta, J_h = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Display gradient descent's result
print('Theta computed from gradient descent:\n Independent term: {},\n ft square coeficient: {},\n # of room coeficient: {}\n'.format(theta[0],theta[1], theta[2]))


# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================

Square_ft = 1650
Num_bedroom = 3      

X_test = np.array((Square_ft, Num_bedroom))
X_test_norm = (X_test - mu)/sigma
X_test_norm = np.hstack((np.ones((1, 1)),X_test_norm))
      
price = np.array(np.dot(X_test_norm, theta)) # Enter your price formula here

# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): {}\n'.format(price))


# Solve with normal equations Analytically:
def normalEqn(X, y):
    #NORMALEQN Computes the closed-form solution to linear regression 
    #   NORMALEQN(X,y) computes the closed-form solution to linear 
    #   regression using the normal equations.
    
    theta = np.zeros((1, X.shape[1]))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the code to compute the closed form solution
    #               to linear regression and put the result in theta.
    #
    
    # ---------------------- Sample Solution ----------------------
    '''
    # theta = ((X.T * X)^-1) * X.T * y
    '''
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)),X.T),y)
    
    # ============================================================
    return theta


# Calculate the parameters from the normal equation
thetaA = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations:\n Independent term: {},\n ft square coeficient: {},\n # of room coeficient: {}\n'.format(thetaA[0],thetaA[1],thetaA[2]))

# Estimate the price of a 1650 sq-ft, 3 br house. 
# ====================== YOUR CODE HERE ======================
Square_ft = 1650
Num_bedroom = 3      
X_test = np.array((Square_ft, Num_bedroom))
X_test = np.hstack((np.ones((1, 1)),X_test.reshape(1,2)))

price = price = np.array(np.dot(X_test_norm, thetaA)) # Enter your price forumla here

# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):{}\n'.format(price)) 