# -*- coding: utf-8 -*-
"""
Applied Mathematics for Computer Science.
Homework1 -- Curve Fitting.

@author:   LiBin 11531041
@date:     2016 - 5 - 23.
"""

import numpy as np
import matplotlib.pyplot as plt


# Function of computing the regression parameters
def Compute_PolyCurFit( x, y, M, par_lambda ):
    x = np.matrix(x)
    y = np.matrix(y)
    M = M + 1                    # 1,x,...,x^M (M+1 features)
    N = np.size(x)
    A = np.matrix( np.zeros((N,M)) )
    
    for i in xrange(M):
        A[:,i] = np.power( x.T, i)
    
    Left_mat  = (A.T) * A + par_lambda * np.matrix( np.eye(M) )
    Right_mat = A.T * y.T
    w = np.linalg.solve(Left_mat, Right_mat)
    return w


# Compute the Polynomial Curve
def Compute_y_bar( x, w, M ):
    x = np.matrix(x)
    M = M + 1
    N = np.size(x)
    A = np.matrix( np.zeros((N,M)) )
    
    for i in xrange(M):
        A[:,i] = np.power( x.T, i)
    
    y_bar = np.array( (A * w).T )
    y_bar = y_bar[0]    # shape transform: N-by-1 Matrix --> N vector.
    return y_bar


# Plot the Figure
def Plot_Figure( Fig_id, M, N, x, y_org, x_sample, y_noise_sample, y_bar, par_lambda ):
    Fig = plt.figure(Fig_id)
    
    plt.plot(x, y_org, color='green',linestyle='solid',linewidth = 2)
    plt.plot(x_sample, y_noise_sample,linestyle='None', marker='o',markerfacecolor='blue', markersize=9)
    plt.plot(x, y_bar, color='red',linestyle='solid',linewidth = 2)
    
    if par_lambda == 0:
        plt.title('Fit degree '+str(M)+' curves in '+str(N)+' samples with lambda = '+str(par_lambda) )
    else:
        plt.title('Fit degree '+str(M)+' curves in '+str(N)+' samples with lambda = '+format(par_lambda,'.3e') )
    
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.xlim( -0.05, 1.05 )
    plt.ylim( -1.5, 1.5 )
    plt.legend(('y=sin(x)','Samples','Polynomial Curve'))

    Fig.show() 
    #Fig.savefig('Homework_1_'+str(Fig_id)+'.pdf')
    #plt.close(Fig)
    return True


# Main Function ( 5 Instance. )

#===============================================================
## 1. sample the function curve of y=sin(x) with Gaussian noise
#     fit degree 3 curves in 10 samples
x = np.linspace(0, 1, 100)
y_org   = np.sin(x*(2*np.pi) )
y_noise = y_org + np.random.randn(100)*0.3

Fig_id = 1
N = 10
M = 3
par_lambda = 0

id_sample_10 = np.int_(np.linspace(0,99,N))
x_sample_10 = x[id_sample_10]
y_noise_sample_10 = y_noise[id_sample_10]

# Compute parameter and test.
w = Compute_PolyCurFit( x_sample_10, y_noise_sample_10, M, par_lambda )
y_bar = Compute_y_bar( x, w, M ) 

Plot_Figure( Fig_id, M, N, x, y_org, x_sample_10, y_noise_sample_10, y_bar, par_lambda )


#==================================================================
## 2. sample the function curve of y=sin(x) with Gaussian noise
#     fit degree 9 curves in 10 samples
Fig_id = 2
N = 10
M = 9
par_lambda = 0

w = Compute_PolyCurFit( x_sample_10, y_noise_sample_10, M, par_lambda )
y_bar = Compute_y_bar( x, w, M )

Plot_Figure( Fig_id, M, N, x, y_org, x_sample_10, y_noise_sample_10, y_bar, par_lambda )

#===================================================================
## 3. sample the function curve of y=sin(x) with Gaussian noise
#     fit degree 9 curves in 15 samples
Fig_id = 3
N = 15
M = 9
par_lambda = 0

id_sample_15 = np.int_(np.linspace(0,99,N))
x_sample_15 = x[id_sample_15]
y_noise_sample_15 = y_noise[id_sample_15]

w = Compute_PolyCurFit( x_sample_15, y_noise_sample_15, M, par_lambda )
y_bar = Compute_y_bar( x, w, M )

Plot_Figure( Fig_id, M, N, x, y_org, x_sample_15, y_noise_sample_15, y_bar, par_lambda )

#=================================================================
## 4. sample the function curve of y=sin(x) with Gaussian noise
#     fit degree 9 curves in 100 samples
Fig_id = 4
N = 100
M = 9
par_lambda = 0

x_sample_100 = x
y_noise_sample_100 = y_noise

w = Compute_PolyCurFit( x_sample_100, y_noise_sample_100, M, par_lambda )
y_bar = Compute_y_bar( x, w, M )

Plot_Figure( Fig_id, M, N, x, y_org, x_sample_100, y_noise_sample_100, y_bar, par_lambda )


#==================================================================
## 5. sample the function curve of y=sin(x) with Gaussian noise
#     fit degree 9 curve in 10 samples but with regularization term
Fig_id = 5
N = 10
M = 9
par_lambda = np.exp(-18)

w = Compute_PolyCurFit( x_sample_10, y_noise_sample_10, M, par_lambda )
y_bar = Compute_y_bar( x, w, M )

Plot_Figure( Fig_id, M, N, x, y_org, x_sample_10, y_noise_sample_10, y_bar, par_lambda )