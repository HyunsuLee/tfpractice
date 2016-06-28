import tensorflow as tf
# https://www.tensorflow.org/versions/r0.9/tutorials/mnist/beginners/index.html#the-mnist-data
#MNIST data import. if you once import data, you don't need import again.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# tf Graph input
x = tf.placeholder(tf.float32, [None, 784], name = "X-input")
y_ = tf.placeholder(tf.float32, [None, 10], name = "Y-input")

#set model weight and bias
W = tf.Variable(tf.zeros([784, 10]), name = "Weight")
b = tf.Variable(tf.zeros([10]), name = "Bias")


#hypothessis
with tf.name_scope("layer") as scope:
  y = tf.nn.softmax(tf.matmul(x, W) + b)

with tf.name_scope("cost") as scope:
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
  cost_sum = tf.scalar_summary("cost", cross_entropy)

with tf.name_scope("train") as scope:
  train = tf.train.GradientDescentOptimizer(0.4).minimize(cross_entropy)


# add histogram
w_hist = tf.histogram_summary("Weight", W)
b_hist = tf.histogram_summary("Bias", b)
y_hist = tf.histogram_summary("Y", y)



init = tf.initialize_all_variables()

with tf.Session() as sess:
  merged = tf.merge_all_summaries()
  writer = tf.train.SummaryWriter("./log/MNIST_logs", sess.graph_def)

#sess = tf.Session()
sess.run(init)

for step in xrange(2000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  summary, _ = sess.run([merged, train], feed_dict={x:batch_xs, y_:batch_ys})
  writer.add_summary(summary, step)

for step in xrange(2000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train, feed_dict={x:batch_xs, y_:batch_ys})
  if step % 10 == 0:
    summary = sess.run(merged, feed_dict={x:batch_xs, y_:batch_ys})
    writer.add_summary(summary, step)


'''
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
'''
