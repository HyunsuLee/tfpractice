import tensorflow as tf

# x1_data = [1, 0, 3, 0, 5]
# x2_data = [0, 2, 0, 4, 0]
#matrix
#remove b
x_data = [[1., 1., 1., 1., 1.,],
          [0., 200., 0., 400., 0.],
          [100., 0., 300., 0., 500.]]
y_data = [100, 200, 300, 400, 500]



#W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
#W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
#matrix
#remove b
W = tf.Variable(tf.random_uniform([1,3], -5.0, 5.0))
# b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))


#hypothesis = W1 * x1_data + W2 * x2_data + b
#matrix, remove b
hypothesis = tf.matmul(W, x_data)

#cost function
cost = tf.reduce_mean(tf.square((hypothesis - y_data)/y_data))


a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)


init = tf.initialize_all_variables()


sess = tf.Session()
sess.run(init)


for step in xrange(2001):
    sess.run(train)
    if step % 50 == 0:
        # print step, sess.run(cost), sess.run(W1), sess(W2), sess.run(b)
        print step, sess.run(cost), sess.run(W) #, sess.run(b)
#gist control.
