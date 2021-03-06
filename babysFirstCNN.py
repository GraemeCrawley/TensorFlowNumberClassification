#Importing tensoflow
import tensorflow as tf
import time as t

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#Starting up a session
sess = tf.InteractiveSession()

#Declaring placeholders (input images and target output classes)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#Creating funtions for weight and bias
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#Convolution and pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#Creating first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#Reshaping x to fit the first layer
x_image = tf.reshape(x, [-1,28,28,1])

#Convolving x_image, add bias, apply ReLU and max pool
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Creating second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#Linking second layer to first
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Adding a fully-connected layer
#1024 is the number of neurons, 7 * 7 * 64 is 7x7 image with 64 features
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

#Linking the layers
#At this point it's pretty clear that the process goes:
#	1. Layer creation
#	2. Connection to previous layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



#A BRIEF MOMENT OF CLARITY
#At this point in writing comments (which yes, should be done while coding, but I'm following a tutorial), I think I know how to make the neural net think that a 2 is a 6. Assuming I can change the training data, I should simply be able to label a 2 as a 6 and feed the net the training data associated with that in order to "train" the neural net to be wrong. Since it has no real concept of what a 2 or 6 are, or even that they're distinct from each other, telling it that a 2 is a 6 seems pretty reasonable.


#Applying a dropout to reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Final layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Training time
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(1000):
  batch = mnist.train.next_batch(10)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#Issue with memory and printing the below line (found out by removing the below line and trying without). Going to try to wait and see if it helps...?
#t.sleep(5)
#Did not work. Will now separate out the line to see where the problem is.
#print("test accuracy %g" %accuracy.eval(feed_dict={
#   x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

#This seems relevant, trying it:
#https://github.com/tensorflow/tensorflow/issues/136


print("Calculating accuracy")

batch_tx, batch_ty = mnist.test.next_batch(10)

accuracy = accuracy.eval(feed_dict={x: batch_tx, y_:batch_ty, keep_prob: 1.0})

print("test accuracy",accuracy)
