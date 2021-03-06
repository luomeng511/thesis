import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# the data
mnist = input_data.read_data_sets('MNIST', one_hot=True)

# Graph input
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# parameters
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# model
y = tf.nn.softmax(tf.matmul(x, W)+b)

# training method
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# run the graph
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={
            x: batch_x,
            y_: batch_y
        })
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={
        x: mnist.test.images,
        y_: mnist.test.labels
    }))
