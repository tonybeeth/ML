
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#matrix of n by 784 with each row consisting of an image of 784 (28x28) pixels
x = tf.placeholder(tf.float32, [None, 784])

#Weights: to be multipled with x to get values for each of the 10 choices
W = tf.Variable(tf.zeros([784, 10]))

#biases to be added
b = tf.Variable(tf.zeros([10]))

#define model
y = tf.matmul(x, W) + b

#Correct(expected) answers
y_ = tf.placeholder(tf.float32, [None, 10])

#using cross entropy to determine loss
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#Above statement is equivalent to
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#Ask tensorflow to minimize the loss using gradient descent algorithm, with learning rate of 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(1000)
    print(type(batch_xs))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_pred = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
