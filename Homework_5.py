# -*- coding: utf-8 -*-
"""
Applied Mathematics for Computer Science.
Homework5 -- 2D SVM.

@author:   LiBin 11531041
@date:     2016 - 5 - 23.
"""
#%%
def gaussian_kernel( x1, x2, delta ):
    result = np.exp(-sum( np.power(x1 - x2, 2) ) / (2* delta*delta) )
    return result

#%%
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers


num_data = 30
x = np.zeros((30,2))
t = np.zeros((30,1))

Fig = plt.figure(1)
row_num = -1
f_r = open('svm_data.txt')
for line in f_r:
    row_num += 1
    seq = line.split()
    #x[row_num,:] = np.array([int(seq[0]), int(seq[1])])
    x[row_num,:] = [float(seq[0]), 450-float(seq[1])]
    t[row_num] = float(seq[2])
    if t[row_num] == 1:
        temp_color = 'red'
    else:
        temp_color = 'blue'
    #plt.plot(x[row_num,0], x[row_num,1],linestyle='None', marker='o',markerfacecolor=temp_color, markersize=9)    

f_r.close()
plt.xlabel('x axis')
plt.ylabel('y axis')
#plt.xlim( 0, 650 )
#plt.ylim( 0, 450 )

mean_x = np.mean( x, axis = 0 )
for i in xrange(num_data):
    x[i,:] = x[i,:] - mean_x
    #x[i,:] = x[i,:] / np.sqrt(sum(np.power(x[i,:],2)))
    if t[i] == 1:
        temp_color = 'red'
    else:
        temp_color = 'blue'
    plt.plot(x[i,0], x[i,1],linestyle='None', marker='o',markerfacecolor=temp_color, markersize=9)    

#%%
# min 1/2 \sum\sum a_n*a_m*t_n*t_m*K(x_n,x_m) - \sum a_n 
#        with constraints a_n >=0, \sum a_n t_n = 0 

par_gaussian_delta = 100

P = np.zeros((num_data,num_data))
for i in xrange(num_data):
    for j in xrange(num_data):
        P[i,j] = 10*t[i,0]*t[j,0]*gaussian_kernel( x[i,:], x[j,:], par_gaussian_delta )
        #P[i,j] = t[i,0]*t[j,0]*( x[i,0]*x[j,0]+x[i,1]*x[j,1])
P = matrix(P)
q = matrix(-np.ones((num_data,1)) )

G = matrix(-np.eye(num_data))
h = matrix(np.zeros((num_data,1)) )

A = matrix(np.ones((1,num_data)) )
b = matrix(np.zeros((1,1)) )

sol=solvers.qp(P, q, G, h, A, b)
a = sol['x']
print 'The solution of a: '
print a
#%%
