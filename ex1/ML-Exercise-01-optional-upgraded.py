# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 12:10:21 2019

@author: LCuretti
"""
import numpy as np
from numpy import loadtxt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
###########################################################################
###############  Optional Exercises: ######################################
###########################################################################
'''

data = loadtxt("ex1data2.txt", comments="#", delimiter=",", unpack=False)
X = data[:, 0:2]
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


Xpoli = X[:]
L = Xpoli[:,1]**2 # num of room ^2
M = Xpoli[:,1]**3 # num of room ^3
O = Xpoli[:,0]*Xpoli[:,1] #Feature 1 * feature 2
Xpoli = np.hstack((Xpoli,L.reshape(m,1)))
Xpoli = np.hstack((Xpoli,M.reshape(m,1)))
Xpoli = np.hstack((Xpoli,O.reshape(m,1)))


Xpoli_norm, mupoli, sigmapoli = featureNormalize(Xpoli)
Xpoli_norm = np.hstack((np.ones((m, 1)),Xpoli_norm))

X_norm, mu, sigma = featureNormalize(X)


# Add intercept term to X
X_norm = np.hstack((np.ones((m, 1)),X_norm))


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
    
    J = sum(((np.dot(X, theta).reshape(m,1))-y.reshape(m,1))**2)/m*.5
    
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
        
        He = (np.dot(X, theta).reshape(m,1))-y.reshape(m,1)
        
        for i in range (0,theta.shape[0]):
            theta[i,0]= theta[i,0] - alpha/m*np.dot(He.T,X[:,i])
      
        # ============================================================
    
        # Save the cost J in every iteration    
        J_history[iter] = computeCostMulti(X, y, theta)

    return theta, J_history

alpha = 0.1
num_iters = 400

# Init Theta and Run Gradient Descent 

theta_X1_norm = np.zeros((2,1))
theta_X1_norm, J_h_X1 = gradientDescentMulti(X_norm[:,0:2], y,theta_X1_norm,alpha, num_iters)

theta_X2_norm = np.zeros((2,1))
theta_X2_norm, J_h_X2 = gradientDescentMulti(X_norm[:,[0,2]], y,theta_X2_norm,alpha, num_iters)

theta_norm = np.zeros((X_norm.shape[1],1))
theta_norm, J_h = gradientDescentMulti(X_norm, y,theta_norm,alpha, num_iters)

# Display gradient descent's result
print('Theta computed from gradient descent:\n Independent term: {},\n ft square coeficient: {},\n # of room coeficient: {}\n'.format(theta_norm[0],theta_norm[1], theta_norm[2]))

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================

Square_ft = 1650
Num_bedroom = 3      

X_test = np.array((Square_ft, Num_bedroom))
X_test_norm = (X_test - mu)/sigma
X_test_norm = np.hstack((np.ones((1, 1)),X_test_norm))
      
price = np.array(np.dot(X_test_norm, theta_norm)) # Enter your price formula here

# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): {}\n'.format(price))


#####################################      
# Denormalization / denormalize of coefficient to be used with raw data and be able to compare with normal equation.    
#####################################
  
def coeff_denormalization(theta,mu,sigma):

    theta_new = np.zeros((theta.shape[0],1))
   
    theta_new[0] = theta[0]
    for i in range (1,theta.shape[0]):
        theta_new[0] = theta_new[0]+theta[i]*(0-mu[0,i-1])/sigma[0,i-1]
        theta_new[i] = theta[i]/sigma[0,i-1]
         
    return theta_new


theta_X1 = coeff_denormalization(theta_X1_norm,mu[0,0].reshape(1,1),sigma[0,0].reshape(1,1))   
theta_X2 = coeff_denormalization(theta_X2_norm,mu[0,1].reshape(1,1),sigma[0,1].reshape(1,1))   #### ver sigma y mu    
theta = coeff_denormalization(theta_norm,mu,sigma)      

X = np.hstack((np.ones((m, 1)),X))

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

price = np.array(np.dot(X_test, thetaA)) # Enter your price forumla here

# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):{}\n'.format(price)) 



Xpoli = np.hstack((np.ones((m, 1)),Xpoli))

thetaP = normalEqn(Xpoli[:,[0,2,3,4]], y)



####################################################### 2 features

xp = np.linspace(500,5000,10)
yp = np.linspace(1,5,10)

Z = np.zeros((len(xp), len(yp)))

for i in range (0,len(xp)):
    for j in range (0,len(yp)):
        Z[i,j]=theta[1]*xp[i]+ theta[2]*yp[j] + theta[0]

Z = Z.T

Xp,Yp = np.meshgrid(xp,yp)

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
J_2f = computeCostMulti(X, y, theta)
ax.scatter(X[:,2], X[:,1], y,'bo')
ax.plot_surface(Yp, Xp, Z)
ax.set_title('2 features with iteration J = {}'.format(J_2f))
ax.set_ylabel('square ft') 
ax.set_xlabel('# bed rooms')
ax.set_zlabel('price')
ax.set_xlim(0, 5)
ax.set_ylim(5000,0)
ax.scatter(0, X[:,1], y, 'rx')
ax.scatter(X[:,2],0, y, 'rx')


Zer = np.zeros((10,1))


ax.plot(Zer,Xp[4,:],Z[4,:],'r-', label = 'projection from midle surface')
ax.plot(Zer,Xp[4,:],(Xp[4,:]*theta_X1[1])+theta_X1[0],'b-', label='regretion 1D')
ax.plot(Yp[:,4],Zer,Z[:,4],'r-')
ax.plot(Yp[:,4],Zer,(Yp[:,4]*theta_X2[1])+theta_X2[0],'b-')

ax.legend()

######################################################################## 4 features Analytically

thetaC = normalEqn(Xpoli[:,0:-1], y)

Zd = np.zeros((len(xp), len(yp)))



for i in range (0,len(xp)):
    for j in range (0,len(yp)):
        Zd[i,j]= thetaC[0] + thetaC[1]*xp[i]+ thetaC[2]*yp[j] + thetaC[3]*yp[j]*yp[j] + thetaC[4]*yp[j]*yp[j]*yp[j]

Zd = Zd.T
J_4fa = computeCostMulti(Xpoli[:,0:-1], y, thetaC)
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.set_title('4 features Analytically J = {}'.format(J_4fa))
ax.scatter(X[:,2], X[:,1], y,'bo')
ax.plot_surface(Yp, Xp, Zd)
ax.set_ylabel('square ft') 
ax.set_xlabel('# bed rooms')
ax.set_zlabel('price')
ax.set_ylim(5000,0)
ax.set_xlim(0, 5)

ax.scatter(0, X[:,1], y, 'rx')
ax.scatter(X[:,2],0, y, 'rx')

Zer = np.zeros((10,1))
ax.plot(Zer,Xp[4,:],(Xp[4,:]*theta_X1[1])+theta_X1[0],'b-', label='regretion #room features Polinomic')
ax.plot(Zer,Xp[4,:],Zd[4,:],'r-', label = 'projection from midle surface')
ax.plot(Yp[:,4],Zer,Zd[:,4],'r-')
ax.plot(Yp[:,4],Zer,Yp[:,4]**3*thetaP[3]+Yp[:,4]**2*thetaP[2]+Yp[:,4]*thetaP[1]+thetaP[0],'b-')
ax.legend()

############################################### 5 features Analytically


thetaB = normalEqn(Xpoli, y)

Zc = np.zeros((len(xp), len(yp)))

 
for i in range (0,len(xp)):
    for j in range (0,len(yp)):
        Zc[i,j]= thetaB[0] + thetaB[1]*xp[i]+ thetaB[2]*yp[j] + thetaB[3]*yp[j]*yp[j] + thetaB[4]*yp[j]*yp[j]*yp[j] + thetaB[5]*xp[i]*yp[j]

Zc = Zc.T
J_5fa = computeCostMulti(Xpoli, y, thetaB)
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.set_title('5 features Analytically  J = {}'.format(J_5fa))
ax.scatter(X[:,2], X[:,1], y,'bo')
ax.plot_surface(Yp, Xp, Zc)
ax.set_ylabel('square ft') 
ax.set_xlabel('# bed rooms')
ax.set_zlabel('price')
ax.set_ylim(5000,0)
ax.set_xlim(0, 5)

ax.scatter(0, X[:,1], y, 'rx')
ax.scatter(X[:,2],0, y, 'rx')

Zer = np.zeros((10,1))
ax.plot(Zer,Xp[4,:],(Xp[4,:]*theta_X1[1])+theta_X1[0],'b-', label='regretion 1D')
ax.plot(Zer,Xp[4,:],Zc[4,:],'r-')
ax.plot(Yp[:,4],Zer,Zc[:,4],'r-')


# Visualizing J_history

index = np.linspace(0,num_iters,num_iters)

fig = plt.figure()
plt.plot(index,J_h, 'b', label = '2 features')
plt.plot(index,J_h_X1, 'y', label = 'Feature 1')
plt.plot(index,J_h_X2, 'g', label = 'Feature 2')
plt.legend()
plt.xlabel('Iteration #')
plt.ylabel('Cost J')

plt.title('Cost vs Iteration')

plt.show()

