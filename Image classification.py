"""
@author: Prasad Gandham
"""
import numpy as np
import os
import pickle
from scipy import ndimage
import time
path='imgs/train'
pixel_depth=255
image_files=os.listdir(path)
image_files.sort()
np.random.seed(255)
dataset=[]
labels=[]
test_dataset=[]
test_labels=[]
y=0
for i in image_files:
    x=0
    for j in os.listdir(path+'/'+i):
	image=(ndimage.imread(path+'/'+i+'/'+j,'L').astype(float)-pixel_depth/2)/pixel_depth
	image=image.ravel()
	#print image
	dataset.append(image)
	labels.append(i)
	x+=1
	if x is 350:
	    y=0
	    test_dataset.append(image)
	    test_labels.append(i)
	    y+=1
	    if y is 100:
	        break


	
	
	    
print 'read data'
k=len(labels)
b=np.zeros((k,10))
b[np.arange(k),labels]=1
print 'read lables'
#print len(labels)
#print len(test_dataset)
k=len(test_labels)
c=np.zeros((k,10))
c[np.arange(k),labels]=1
del labels
del test_labels

import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 2560 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 307200 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        #total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        _, c = sess.run([optimizer, cost], feed_dict={x: dataset,y: b})
            # Compute average loss
        avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost)
    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: test_dataset, y: c})
