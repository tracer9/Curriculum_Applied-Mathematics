# -*- coding: utf-8 -*-
"""
Applied Mathematics for Computer Science.
Homework3 -- 2D-MOG and  K-means.

@author:   LiBin 11531041
@date:     2016 - 5 - 23.
"""
import numpy as np
import matplotlib.pyplot as plt

#%% gaussian pdf
def gaussian_pdf( x, mean, cov ):
    #cov = cov + 0.2*np.eye(2)
    t  = x - mean
    t1 = np.exp( -0.5* np.dot( np.dot( t, np.linalg.inv(cov) )  , t.reshape(2,1) ) )
    t2 = np.sqrt( np.linalg.det(cov) )
    p  = t1 / t2
    return p
    

#%% generate the original data
K = 3
mean = np.random.rand(2,K)*8
cov  = np.random.rand(2,K)+0.5

#print mean
#print cov

#num_sample = 500
data = np.zeros((1500,2))

data[0:400,0],data[0:400,1] = np.random.multivariate_normal(mean[:,0] ,np.diag(cov[:,0]) ,400).T
data[400:900,0],data[400:900,1] = np.random.multivariate_normal(mean[:,1] ,np.diag(cov[:,1]) ,500).T
data[900:1500,0],data[900:1500,1] = np.random.multivariate_normal(mean[:,2] ,np.diag(cov[:,2]) ,600).T

#plt.plot(data[0:400,0],data[0:400,1],'rx')
#plt.plot(data[400:900,0],data[400:900,1],'gx')
#plt.plot(data[900:1500,0],data[900:1500,1],'bx')
max_picture = 8
row = 2

t = int(str(row)+str(max_picture/row)+'1')
plt.subplot(t)
plt.plot(data[0:400,0],data[0:400,1],'rx')
plt.plot(data[400:900,0],data[400:900,1],'gx')
plt.plot(data[900:1500,0],data[900:1500,1],'bx')
plt.title('Ground Truth')

t = int(str(row)+str(max_picture/row)+'2')
plt.subplot(t)
plt.plot(data[:,0],data[:,1],'kx')
plt.title('Original Data')


#%% -------------
# greate the random index
color_list = ['r', 'g', 'b']

#original_index = np.random.permutation(1500)
#
membership_matrix = np.zeros((1500,3))
#membership_matrix[original_index[0:500],0] = 1
#membership_matrix[original_index[500:1000],1] = 1
#membership_matrix[original_index[1000:1500],2] = 1
#
#index_picture = int(str(row)+str(max_picture/row)+'2')
#plt.subplot(index_picture)
#plt.plot( data[original_index[0:500],0], data[original_index[0:500],1], 'rx' )
#plt.plot( data[original_index[500:1000],0], data[original_index[500:1000],1], 'gx' )
#plt.plot( data[original_index[1000:1500],0], data[original_index[1000:1500],1], 'bx' )


#%%-------------------------
Num_cluster = np.zeros((1,3))
pi_cluster = np.zeros((1,3))
mean_cluster = np.zeros((3,2))
cov_cluster = np.zeros((3,2,2))

mean_cluster[0,:] = data[0,:]
mean_cluster[1,:] = data[400,:]
mean_cluster[2,:] = data[900,:]
cov_cluster[0,:,:] = np.eye(2)
cov_cluster[1,:,:] = np.eye(2)
cov_cluster[2,:,:] = np.eye(2)


for loop in xrange(max_picture-2):

# E-step
    for i in xrange(1500):
        for k in xrange(3):
            membership_matrix[i,k] = gaussian_pdf( data[i,:], mean_cluster[k,:], cov_cluster[k,:,:] )
            #membership_matrix[i,k] = pi_cluster[0,k]*gaussian_pdf( data[i,:], mean_cluster[k,:], cov_cluster[k,:,:] )
        membership_matrix[i,:] = membership_matrix[i,:] / sum(membership_matrix[i,:])

    membership_max = np.amax( membership_matrix, axis = 1 )
    membership_index = np.zeros((1500,1))
    for i in xrange(1500):
        membership_index[i] = list(membership_matrix[i,:]).index( membership_max[i] )
        
    index_picture = int(str(row)+str(max_picture/row)+str(loop+3))
    plt.subplot(index_picture)
    plt.title('Step-'+str(loop+1))
    for k in xrange(3):
        t = ( membership_index == k )
        t = t[:,0]  
        temp_data = data[t, :]
        color = color_list[k]+'x'
        plt.plot(temp_data[:,0],temp_data[:,1],color)
        
#M-step
    for k in xrange(3):
        Num_cluster[0,k] = sum( membership_matrix[:,k])
        pi_cluster[0,k]  = Num_cluster[0,k] / 1500
        mean_cluster[k,:] = np.dot(membership_matrix[:,k].reshape(1,1500) , data) / Num_cluster[0,k]
        for i in xrange(1500):
            t = data[i,:] - mean_cluster[k,:]
            cov_cluster[k,:,:] = cov_cluster[k,:,:] + membership_matrix[i,k] * ( t.reshape(2,1) * t )
        cov_cluster[k,:,:] = cov_cluster[k,:,:] / Num_cluster[0,k]

#print 'original parameters'
#print mean
#print cov
#
#print 'result parameters'
#print mean_cluster
#print cov_cluster

        
#    Num_0 = sum( membership_matrix[:,0])
#    pi_0 = Num_0 / 1500
#    mean_0 = (membership_matrix[:,0].reshape(1,1500) * data) / Num_0
#    cov_0 = np.zeros((2,2))
#    for i in xrange(1500):
#        t = data[i,:] - mean_0
#        cov_0 = cov_0 + membership_matrix[i,0] *( t.reshape(2,1) * t )
#    cov_0 = cov_0 / Num_0
#    ###------------------------------------
#    ###------------------------------------
#    Num_1 = sum( membership_matrix[:,1])
#    pi_1 = Num_1 / 1500
#    mean_1 = (membership_matrix[:,1].reshape(1,1500) * data) / Num_1
#    cov_1 = np.zeros((2,2))
#    for i in xrange(1500):
#        t = data[i,:] - mean_1
#        cov_1 = cov_1 + membership_matrix[i,1] *( t.reshape(2,1) * t )
#    cov_1 = cov_1 / Num_1
#    ###------------------------------------
#    ###------------------------------------
#    Num_2 = sum( membership_matrix[:,2])
#    pi_2 = Num_2 / 1500
#    mean_2 = (membership_matrix[:,2].reshape(1,1500) * data) / Num_1
#    cov_2 = np.zeros((2,2))
#    for i in xrange(1500):
#        t = data[i,:] - mean_2
#        cov_2 = cov_2 + membership_matrix[i,2] *( t.reshape(2,1) * t )
#    cov_2 = cov_2 / Num_2
###------------------------------------
###------------------------------------
