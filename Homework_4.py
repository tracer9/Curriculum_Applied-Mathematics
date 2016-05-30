# -*- coding: utf-8 -*-
"""
Applied Mathematics for Computer Science.
Homework4 -- L-M Algorithm.

@author:   LiBin 11531041
@date:     2016 - 5 - 23.
"""

#%% Objective: Assuming given type of the certain function 
#   " fun(x) = a*exp(-b*t) ", input data "x1,...x10", and output data "y1,..y10",
#   using the Levenberg-Marquardt algorithm to find out the optimial value of 
#   "a" and "b". Naturally, the objective function is f(x) = = 1/2 * sum( ( fun(x_i)-y_i) * 2)

#%%
#0.1 compute the F_x, where F_i(x) = a*exp(-b *x_i) - y_i
def F_x( x, y, a, b ):
    result = (a* np.exp(-b * x) - y).T
    return result

#0.2 compute the jacobian matrix
def J_x( x, a, b ):
    result = np.matrix(np.zeros((10,2)) )
    result[:,0] = np.exp(-b*x).T
    result[:,1] = np.multiply(-(a*x), np.exp(-b*x) ).T
    return result

#0.3 compute the f_x, where f(x) = 1/2 * sum( F_x .* 2)
def f_x( x, y, a, b ):
    temp = a* np.exp(-b * x) - y
    result = np.sum( np.power( temp, 2) ) /2
    return result
    
#%%
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

x = np.matrix([0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10])
y = np.matrix([19.21, 18.15, 15.36, 14.10, 12.89, 9.32, 7.45, 5.24, 3.01, 1.85])


mu = 0.01
epsilon = 1e-6
max_iter = 50

a=10
b=0.5


#%%
a_trend = []
b_trend = []
f_trend = []

for loop in range(max_iter):
    J = J_x( x, a, b )
    F = F_x( x, y, a, b )
## step - 2    
    g = J.T * F
    G = J.T * J
## step - 3
    norm_g = np.sqrt( sum( np.power( g, 2) ) )
    if norm_g < epsilon:
        break
## step - 4
    key = 0
    while key == 0:
        G_mu = G + mu * np.eye(2)
        if np.all( np.linalg.eigvals(G_mu)>0 ):
            key = 1
        else:
            mu = 4 * mu
            key = 0
## step - 5
    s = np.linalg.solve( G_mu, -g )

## step - 6
    a_new = a + s[0,0]
    b_new = b + s[1,0]
    diff_f = f_x( x, y, a_new, b_new ) - f_x( x, y, a, b )
    diff_q = (J.T * F).T * s + (s.T*(J.T*J) *s) /2
    r = diff_f / diff_q
## step - 7
    if r < 0.25:
        mu = mu * 4
    elif r > 0.75:
        mu = mu / 2
    else:
        pass
## step - 8
    if r > 0:
        a = a_new
        b = b_new
    else:
        pass
    #print mu
    a_trend.append(a)
    b_trend.append(b)
    f_trend.append(np.log(f_x( x, y, a, b)) )

#%%

num_grid = 15
a_index,b_index = np.mgrid[5:25:num_grid*1j,0:0.5:num_grid*1j]
z = np.zeros((num_grid,num_grid))
for i in xrange(num_grid):
    for j in xrange(num_grid):
        z[i,j] = np.log( f_x( x, y, a_index[i,j], b_index[i,j] ) )

ax = plt.subplot(111,projection='3d')
ax.plot_surface(a_index,b_index,z,rstride=2,cstride=1,cmap=plt.cm.coolwarm,alpha=0.8)
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('log f(x)')

mpl.rcParams['legend.fontsize'] = 10
ax.plot(a_trend, b_trend, f_trend, color='blue',linestyle='solid',linewidth = 3,marker='o',markerfacecolor='red',markersize=9,label='optimization curve')
ax.legend(loc=3)
plt.title('L-M algorithm to evaluate the optimial value')

plt.show()
