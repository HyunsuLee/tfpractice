#Multi GPU Basic example
'''
This tutorial requires your machine to have 2 GPUs
"/cpu:0": The CPU of your machine.
"/gpu:0": The first GPU of your machine
"/gpu:1": The second GPU of your machine
'''

import numpy as np
import tensorflow as tf
import datetime

#Processing Units logs
log_device_placement = True

#num of multiplications to perform
n = 40

'''
Example: compute A^4 + B^4 on 2 GPUs
Results on 8 cores with GTX-1080 and GTX-960:
 * Single GPU computation time: 0:56:24.982570
 * Multi GPU computation time:  0:54:28.246102

Example: compute A^3 + B^3 on 1 GPUs
Results on 8 cores with GTX-1080 and GTX-960:
 * GTX1080 GPU computation time: 0:00:03.4~5
 * GTX960 GPU computation time:  0:00:03.4~5
'''
#Create random large matrix
A = np.random.rand(1e3, 1e3).astype('float32')
B = np.random.rand(1e3, 1e3).astype('float32')

# Creates a graph to store results
c1 = []
c2 = []

def matpow(M, n):
    if n < 1: #Abstract cases where n < 1
        return M
    else:
        return tf.matmul(M, matpow(M, n-1))

'''
Single GPU computing
'''
with tf.device('/gpu:1'):
    a = tf.constant(A)
    b = tf.constant(B)
    #compute A^n and B^n and store results in c1
    c1.append(matpow(a, n))
    c1.append(matpow(b, n))

with tf.device('/gpu:1'):
    sum = tf.add_n(c1) #Addition of all elements in c1, i.e. A^n + B^n

t1_1 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    # Runs the op.
    sess.run(sum)
t2_1 = datetime.datetime.now()


'''
Multi cPU computing
'''
#GPU:0 computes A^n
with tf.device('/cpu:0'):
    #compute A^n and store result in c2
    a = tf.constant(A)
    c2.append(matpow(a, n))

#GPU:1 computes B^n
with tf.device('/cpu:0'):
    #compute B^n and store result in c2
    b = tf.constant(B)
    c2.append(matpow(b, n))

with tf.device('/cpu:0'):
  sum = tf.add_n(c2) #Addition of all elements in c2, i.e. A^n + B^n

t1_2 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    # Runs the op.
    sess.run(sum)
t2_2 = datetime.datetime.now()


print ("Single GPU computation time: " + str(t2_1-t1_1))
print ("Multi GPU computation time: " + str(t2_2-t1_2))