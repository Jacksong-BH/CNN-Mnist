
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from collections import Counter

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")


# In[2]:


from sklearn.preprocessing import MinMaxScaler
NormalizeScaler = MinMaxScaler()


# In[3]:


#import mnist datasets
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)


# In[4]:


mnist_train_features = NormalizeScaler.fit_transform(mnist.train.images)
mnist_valid_features = NormalizeScaler.fit_transform(mnist.validation.images)
mnist_test_features = NormalizeScaler.fit_transform(mnist.test.images)


# In[28]:


#define parameters
learning_rate = 0.01
epochs = 2
n_classes = 10  #Labels number
batch_size = 200
validate_size = 2000


# In[6]:


#define network inputs and outputs 
x = tf.placeholder(tf.float32,[None,28*28])
y_ = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32,name='keep_prob')


# In[7]:


#define CNN parameters, FC layer and output layer
weights = {'wc1':tf.Variable(tf.random_normal([5,5,1,32],dtype=tf.float32)),
          'wc2':tf.Variable(tf.random_normal([5,5,32,64],dtype=tf.float32)),
          'wd1':tf.Variable(tf.random_normal([7*7*64,1024],dtype=tf.float32)),
          'out':tf.Variable(tf.random_normal([1024,n_classes],dtype=tf.float32))
          }

biases = {'bc1':tf.Variable(tf.random_normal([32],dtype=tf.float32)),
         'bc2':tf.Variable(tf.random_normal([64],dtype=tf.float32)),
         'bd1':tf.Variable(tf.random_normal([1024],dtype=tf.float32)),
         'out':tf.Variable(tf.random_normal([n_classes],dtype=tf.float32))
         }


# In[8]:


#General function for getting data batch
def get_batches(features, labels, batch_size):
    if len(features) == len(labels):
        for idx in range(0,len(features),batch_size):
            if (idx+batch_size) <= len(features):
                x = features[idx:idx+batch_size]
                y_ = labels[idx:idx+batch_size]
                yield x,y_
            else:
                x = features[idx:]
                y_ = labels[idx:]
                yield x,y_
    else:
          print("Data length error, length of features and labels are not equal.")  


# In[9]:


#define CNN framework
def conv2d(x,w,b,strides):
    x = tf.nn.conv2d(x,filter=w,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,bias=b)
    return tf.nn.relu(x)

def maxpool2d(x,k):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

def conv_net(x,weights,biases,dropout):
    #first convolution layer x*28*28*1->x*14*14*32
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], strides = 1)
    conv1 = maxpool2d(conv1, k = 2)
    #second convolution layer x*14*14*32->x*7*7*64
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], strides = 1)
    conv2 = maxpool2d(conv2, k= 2)
    #full connection layer x*7*7*64->x*1024
    fc1 = tf.reshape(conv2,shape=[-1,weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    
    #output layer
    out = tf.add(tf.matmul(fc1,weights['out']), biases['out'])
    return out


# In[10]:


#reshape input data to x*28*28*1
x_reshape = tf.reshape(x,[-1,28,28,1])
#CNN model
logits = conv_net(x_reshape,weights,biases,keep_prob)
logits_softmax = tf.nn.softmax(logits)


# In[11]:


#loss, cost and optimizor
#cost = -1 * tf.reduce_sum(tf.multiply(y_,tf.log(logits)))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(logits_softmax,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,dtype=tf.float32))


# In[12]:


#initilize variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()


# In[29]:


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for batch in range(mnist.train.num_examples//batch_size):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _,loss = sess.run([optimizer,cost], feed_dict = {x: batch_x, y_: batch_y, keep_prob: 0.4})
            
            validate_batch_x, validate_batch_y = mnist.validation.next_batch(validate_size)
            validation_accuracy = sess.run(accuracy, feed_dict = {x: validate_batch_x, y_: validate_batch_y, keep_prob: 1.})
            print('Epoch {:>2}, Batch {:>3} -'
                 'Loss: {:.4f}, Validation accuracy: {:.6f}'.format(epoch+1, batch+1, loss, validation_accuracy))
    
    #Test accuracy
    test_acc = sess.run(accuracy, feed_dict = {x: mnist.test.images[:3000], y_: mnist.test.labels[:3000], keep_prob: 1.})
    print('Testing data accuracy: {}'.format(test_acc))


# In[13]:


with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(epochs):
        counter = 0
        for batch_x, batch_y in get_batches(mnist_train_features, mnist.train.labels, batch_size):
            _,loss = sess.run([optimizer,cost], feed_dict = {x: batch_x, y_: batch_y, keep_prob: 0.6})
            
            counter += 1
            #validate_batch_x, validate_batch_y = mnist.validation.next_batch(validate_size)
            validation_accuracy = sess.run(accuracy, feed_dict = {x: mnist_valid_features[:validate_size], y_: mnist.validation.labels[:validate_size], keep_prob: 1.})
            print('Epoch {:>2}, Batch {:>3} -'
                 'Loss: {:.4f}, Validation accuracy: {:.6f}'.format(epoch+1, counter, loss, validation_accuracy))
    
    #Test accuracy
    test_acc = sess.run(accuracy, feed_dict = {x: mnist_test_features[:3000], y_: mnist.test.labels[:3000], keep_prob: 1.})
    print('Testing data accuracy: {}'.format(test_acc))

