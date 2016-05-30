# -*- coding: utf-8 -*-
"""
Applied Mathematics for Computer Science.
Homework5 -- SVM.

@author:   LiBin 11531041
@date:     2016 - 5 - 23.
"""


import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import matplotlib.pyplot as plt


# Utill Functions.

def linear_kernel(x1, x2):
    return np.dot(x1,x2)

def kernel_matrix(x, n_samples):
    kernel = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            kernel[i,j] = linear_kernel(x[i],x[j])
    return kernel

def f(x,w,b, c=0):
    return (-w[0]*x -b + c) / w[1]


# Generate 2D data
mean1 = np.array([0,2.5])
mean2 = np.array([2.5,0])
cov = np.array([[0.4,0.6],[0.6,0.4]])

# Positive Samples.
X1 = np.random.multivariate_normal(mean1, cov, 100)
y1 = np.ones(len(X1))
# Negative Samples.
X2 = np.random.multivariate_normal(mean2, cov, 100)
y2 = np.ones(len(X2)) * (-1)

# Split for training and testing data.

X1_train = X1[:90]
y1_train = y1[:90]
X2_train = X2[:90]
y2_train = y2[:90]

X1_test  = X1[90:]
y1_test  = y1[90:]
X2_test  = X2[90:]
y2_test  = y2[90:]

X_train = np.vstack( (X1_train, X2_train) )
y_train = np.hstack( (y1_train, y2_train) )
X_test  = np.vstack( (X1_test, X2_test) )
y_test  = np.hstack( (y1_test, y2_test) )

sample = len(X_train)
K = kernel_matrix(X_train, len(X_train))

# Plot data.
Fig1 = plt.figure(1)

for i in range(100):
    plt.plot(X1[i,0], X1[i,1],linestyle='None', marker='o',markerfacecolor='red', markersize=9)
for i in range(100):
    plt.plot(X2[i,0], X2[i,1],linestyle='None', marker='o',markerfacecolor='blue',markersize=9)

plt.title('Original Data.')
    
plt.savefig("./svm1.png",dpi=72)
Fig1.show()


# CVXOPT -- QP

# objective function
P = cvxopt.matrix(np.outer(y_train, y_train) * kernel_matrix(X_train, 180) )
q = cvxopt.matrix(np.ones(sample) * (-1) )
# equation constraint
A = cvxopt.matrix(y_train,(1,sample))
b = cvxopt.matrix(0.0)
# inequation constraint
G = cvxopt.matrix(np.diag(np.ones(sample) * -1))
h = cvxopt.matrix(np.zeros(sample))

# Solve. a should be the lagrangian multiplier.

solution = cvxopt.solvers.qp(P, q, G, h, A, b)
a = np.ravel(solution['x'])     # a is lagrangian coefficient.


# Transform lagrangian coefficient to w and b.

# support vector.

sv = a > 1e-5
ind = np.arange(len(a))[sv]

a_sv = a[sv]
x_sv = X_train[sv]
y_sv = y_train[sv]

# Solving b.
b = 0
for n in range(len(a_sv)):
    b += y_sv[n]
    b -= np.sum(a_sv * y_sv * K[ind[n], sv])
b /= len(a_sv)

# Solving w
w = np.zeros(2)
for n in range(len(a_sv)):
    w += a_sv[n] * y_sv[n] * x_sv[n]

# Plot2.

Fig2 = plt.figure(2)

a0 = -2
a1 = 4
b0 = f(a0, w, b)
b1 = f(a1, w, b)

plt.plot([a0,a1], [b0,b1], 'k--')
plt.plot([a0,a1], [f(a0,w,b,1),f(a1,w,b,1)], 'k--')
plt.plot([a0,a1], [f(a0,w,b,-1),f(a1,w,b,-1)], 'k--')

plt.title('SVM model.')

for i in range(len(X1_train)):
    plt.plot(X1[i,0], X1[i,1],linestyle='None', marker='o',markerfacecolor='red', markersize=9)
for i in range(len(X2_train)):
    plt.plot(X2[i,0], X2[i,1],linestyle='None', marker='o',markerfacecolor='blue',markersize=9)

Fig2.show()

# Test 

res1 = np.sign(np.dot(X1_test, w) + b)
res2 = np.sign(np.dot(X2_test, w) + b)

res = np.hstack( (res1,res2) )

accuracy = np.sum(res == y_test) / len(y_test)
