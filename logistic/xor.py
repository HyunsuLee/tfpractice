import tensorflow as tf
import numpy as np

xy = np.loadtxt('xor.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32, name = "X-input")
Y = tf.placeholder(tf.float32, name = "Y-input")

W1 = tf.Variable(tf.random_uniform([2,2], -1.0, 1.0), name = "Weight1")
W2 = tf.Variable(tf.random_uniform([2,1], -1.0, 1.0), name = "Weight2")

b1 = tf.Variable(tf.zeros([2]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")

with tf.name_scope("layer2") as scope:
    L2 = tf.sigmoid(tf.matmul(W1, X) + b1)

with tf.name_scope("layer3") as scope:
    hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)

with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
    cost_summ = tf.scalar_summary("cost", cost)

with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(cost)

# add histogram
w1_hist = tf.histogram_summary("Weight1", W1)
w2_hist = tf.histogram_summary("Weight2", W2)

b1_hist = tf.histogram_summary("Bias", b1)
b2_hist = tf.histogram_summary("Bias", b2)

y_hist = tf.histogram_summary("Y", Y)


init = tf.initialize_all_variables()

with tf.Session() as sess:
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./log/xor_logs", sess.graph_def)

sess = tf.Session()
sess.run(init)
'''
for step in xrange(200000):
  summary, _ = sess.run([merged, train], feed_dict={X:x_data, Y:y_data})
  writer.add_summary(summary, step)
'''

for step in xrange(200000):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 2000 == 0:
        summary = sess.run(merged, feed_dict={X:x_data, Y:y_data})
        writer.add_summary(summary, step)

