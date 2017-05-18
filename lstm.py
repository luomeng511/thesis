import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import rnn_cell
from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.models.rnn import rnn_cell


# This is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Hyperparameters
lr = 0.001                              # learning rate
trainning_iters = 10000                 # iteration times
batch_size = 128                        #

n_inputs = 28                           # the shape of image is 28*28 (every time input 28 data point)
n_steps = 28                            # time steps (equal to the row of image)
n_hidden_units = 128                    # neurons in hidden layer
n_class = 10                            # MNIST classes (0-9)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_class])

# Define weights and biases
weights = {
    # (28,128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128,28)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_inputs]))
}
biases = {
    # (128,1)
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10,)
    'out': tf.Variable(tf.constant(0.1, shape=[n_class, ]))
}


def RNN(X, weights, biases):
    # hidden layer for input to cell
    # X(128 batch, 28 steps, 28 inputs) --> X(128*28, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    # X_in --> (128 batchs * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in --> (128 batchs, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])


    # cell

    lstm_cell = rnn_cell.CoupledInputForgetGateLSTMCell(n_hidden_units, forget_bias=1.0)
    # lstm is divided into two parts(c_state, m_state)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)       # time_major means time step in the first dimension



    # hidden layer for output as the final results
    results = tf.matmul(states[1], weights['out']) + biases['out']


    return results



prediton = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediton, y))   #####
train_op = tf.train.AdamOptimizer(lr).minimize(cost)   ####

correct_prediton = tf.equal(tf.argmax(prediton, 1), tf.argmax(y, 1))   ###
accuracy = tf.reduce_mean(tf.cast(correct_prediton, tf.float32))    #####

if __name__ == '__main__':
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        step = 0
        while step * batch_size < trainning_iters:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
            sess.run([train_op], feed_dict={
                x: batch_xs,
                y: batch_ys
            })
            if step % 20 == 0:
                print(sess.run(accuracy, feed_dict={
                    x: batch_xs,
                    y: batch_ys
                }))

