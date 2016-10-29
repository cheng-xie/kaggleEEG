import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import sys

from utils import EEGDataLoader

learning_rate = 0.00006
training_iters = 100000
batch_size = 64 
display_step = 4

n_channels = 16
n_hidden = 128 
n_classes = 2

n_steps = 2048 

x = tf.placeholder("float", [batch_size, n_steps, n_channels])
y = tf.placeholder("float", [batch_size, n_classes])

weights = {
    'out': tf.Variable( tf.random_normal([2*n_hidden, n_classes]) )  
}

biases = {
    'out': tf.Variable( tf.random_normal([n_classes]) )
}

def BiRNN(x, weights, biases):
    # shape of x will be (batch_size, n_steps, n_channels)
    # need to transform to (n_steps, batch_size, n_channels) to comply with tf bi_rnn
    x = tf.transpose(x, [1, 0, 2])
    x = tf.unpack(x, axis = 0)

    # Forward direction cell
    lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    try:
        outputs, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype = tf.float32)
    except Exception:
        outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype = tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = BiRNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# Evaluate Model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4)) as sess:
    stride = n_steps
    trainf = sys.argv[1]
    testf = ''
    print("Loading data files")
    dataloader = EEGDataLoader(trainf, testf, batch_size, n_steps, stride)  
    sess.run(init)
    step = 1
    ma_acc = 0
    ma_loss = 0

    while step * batch_size < training_iters:
        batch_x, batch_y = dataloader.next()
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            ma_acc = acc*0.25 + ma_acc*0.75

            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            ma_loss = loss*0.25 + ma_loss*0.75

            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc) + ", MA Loss= " + \
                  "{:.6f}".format(ma_loss) + ", MA Training= " + \
                  "{:.5f}".format(ma_acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    '''
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
    '''
