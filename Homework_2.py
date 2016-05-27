# -*- coding: utf-8 -*-
"""
Applied Mathematics for Computer Science.
Homework2 -- PCA.

@author:   LiBin 11531041
@date:     2016 - 5 - 23.
"""


import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
#import PIL.Image as Im

def str2vector( vector, N ):
    res = np.zeros( ( N, 1) )
    for i in xrange( N ):
        res[i] = int( vector[i])
    return res


digit = '3'
row_id = 0
picture_row = []


# Using "with" do data IO is better
# After this iteration, we got "picture row" -- each row number for "3".
f_r = open('optdigits-orig-tra.txt','r')
for line in f_r:
    row_id += 1
    seq = line.split()
    if seq[0] == digit:
        picture_row.append(row_id-32)
f_r.close()       


# build the digit matrix D*N (D is 32^2=1024, N is sample numbers.)
num_picture = len(picture_row)
digit_matrix = np.matrix( np.zeros( (32**2, num_picture ) ) )

row_id = 0
key = 0
column = -1
f_r = open('optdigits-orig-tra.txt','r')
for line in f_r:
    row_id += 1
    if row_id in picture_row:
        column = column +1
        key = 32
    if key > 0:
        seq = line.split()
        vector = str2vector( seq[0], 32 )
        start_id = (32-key)*32
        end_id   = (33-key)*32
        digit_matrix[start_id:end_id,column] = vector
        key = key - 1
f_r.close()

# Compute the mean.
x_bar_vector = np.mean( digit_matrix, axis=1)

#%%
digit_matrix_no_center = np.array( digit_matrix - np.tile(x_bar_vector, num_picture ))

U, s, Vh = svd( digit_matrix_no_center )
V = Vh.T
first_PC  = V[:,0]
second_PC = V[:,1]

#%%
plt.subplot(121)
plt.plot( first_PC, second_PC, linestyle='None', marker='o',markerfacecolor='green', markersize=8)

plt.plot( [-0.22,0.22],[-0.12,-0.12],color='black', linestyle='--' )
plt.plot( [-0.22,0.22],[-0.06,-0.06],color='black', linestyle='--' )
plt.plot( [-0.22,0.22],[0,0],color='black', linestyle='--' )
plt.plot( [-0.22,0.22],[0.06,0.06],color='black', linestyle='--' )
plt.plot( [-0.22,0.22],[0.12,0.12],color='black', linestyle='--' )

plt.plot( [-0.12,-0.12],[-0.22,0.22],color='black', linestyle='--' )
plt.plot( [-0.06,-0.06],[-0.22,0.22],color='black', linestyle='--' )
plt.plot( [0,0],[-0.22,0.22],color='black', linestyle='--' )
plt.plot( [0.06,0.06],[-0.22,0.22],color='black', linestyle='--' )
plt.plot( [0.12,0.12],[-0.22,0.22],color='black', linestyle='--' )

plt.xlabel('x axis')
plt.ylabel('y axis')
plt.xlim( -0.2, 0.2 )
plt.ylim( -0.2, 0.2 )

#%%
label_address = np.zeros( (25,2) )
for i in xrange(25):
    row_id    = i // 5
    column_id = i %  5 
    label_address[i,:] = [(row_id-2)*0.06, (column_id-2)*0.06 ]
        
#%%
        
distance_matrix = np.zeros( (25, num_picture ) )
for i in xrange(25):
    for j in xrange(num_picture):
        distance_matrix[i,j] = np.sqrt( (label_address[i,0]- V[j,0])**2 + (label_address[i,1]- V[j,1])**2 )

distance_min = np.amin( distance_matrix, axis = 1 )
distance_address = np.zeros(25)
for i in xrange(25):
    distance_address[i] = list(distance_matrix[i,:]).index( distance_min[i] )

#%%
x_redpoint = V[np.int_(distance_address),0]
y_redpoint = V[np.int_(distance_address),1]
plt.plot(x_redpoint, y_redpoint, 'ro')


#%%
image_25_matrix = np.matrix( np.zeros( (32*5, 32*5 ) ) )
for i in xrange(5):
    for j in xrange(5):
        x_start_id = (4-i)*32
        x_end_id   = (5-i)*32 
        y_start_id = j*32 
        y_end_id   = (j+1)*32
        image_25_id = i*5+j
        image_temp = digit_matrix[:, distance_address[image_25_id] ]
        image_25_matrix[x_start_id:x_end_id, y_start_id:y_end_id] = image_temp.reshape(32,32)

#%%
x_index = []
y_index = []
for i in xrange(32*5):
    for j in xrange(32*5):
        if image_25_matrix[i,j] == 1:
            x_index.append(j)
            y_index.append(159-i)
plt.subplot(122)
plt.plot(x_index, y_index, 'ko', markersize= 3)

#image_25_picture = Im.fromarray( np.uint8( 255 - image_25_matrix * 255 ) )
#image_25_picture.show()
#%%
#u_1 = U[:,1]
#u_1_s = ( u_1 - min(u_1) )
#u_1_image = Im.fromarray( np.uint8( 255 - u_1_s.reshape(32,32) * 255 ) )
##u_1_image.show()
#title = 'digit_'+ digit + '_First_Principle_Component.bmp'
#u_1_image.save( title )

#u_2 = U[:,2]
#u_2_s = ( u_1 - min(u_2) )
#u_2_image = Im.fromarray( np.uint8( 255 - u_2_s.reshape(32,32) * 255 ) )
##u_1_image.show()
#title = 'digit_'+ digit + '_Second_Principle_Component.bmp'
#u_2_image.save( title )