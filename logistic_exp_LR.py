#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:47:02 2017

@author: ravi
"""


import time
import numpy as np
import tensorflow as tf
import random


MAX_WORDS = 10000
#NO_LABLES = 49
#L_RATE = 1
TRAINING_EPOCHS = 30
BATCH_SIZE = 64
BETA = 0.001
DECAY_RATE = 0.80
INR_RATE = 1.8
START_LRATE = float(1)




train_mat = np.load("/home/dravishankar7/mllds/assign2/savedMatrix/f_train_mat.npy")
train_lab_onehot = np.load("/home/dravishankar7/mllds/assign2/savedMatrix/f_train_lab_onehot.npy")
#dev_mat = np.load("/home/dravishankar7/mllds/assign2/savedMatrix/f_dev_mat.npy")
#dev_lab_onehot = np.load("/home/dravishankar7/mllds/assign2/savedMatrix/f_dev_lab_onehot.npy")

'''
train_mat = np.load("/home/ravi/Documents/MLLDS/Assignment2/savedMatrix/train_mat.npy")
train_lab_onehot = np.load("/home/ravi/Documents/MLLDS/Assignment2/savedMatrix/train_lab_onehot.npy")
#dev_mat = np.load("/home/ravi/Documents/MLLDS/Assignment2/savedMatrix/dev_mat.npy")
#dev_lab_onehot = np.load("/home/ravi/Documents/MLLDS/Assignment2/savedMatrix/dev_lab_onehot.npy")
'''

NO_LABLES = len(train_lab_onehot[0])

print("loaded data")

def batchGen(train_mat, train_lab_onehot):
    data_size = len(train_mat)
    for b_epoch in range(TRAINING_EPOCHS):
        for b_nobatches in range(NO_BATCHES):
            samp = random.sample(range(data_size), BATCH_SIZE)
            samp_data = train_mat[samp]
            samp_lab = train_lab_onehot[samp]
            #print("shape of samp_data: ", samp_data.shape)
            #print("shape of samp_lab: ", samp_lab.shape)
            yield samp_data, samp_lab


x = tf.placeholder(tf.float32,[None, MAX_WORDS])
y = tf.placeholder(tf.float32,[None, NO_LABLES])


w = tf.Variable(tf.random_normal(shape=[MAX_WORDS, NO_LABLES], mean=0.0,stddev=1.0))
b = tf.Variable(tf.random_normal(shape=[NO_LABLES], mean=0, stddev=1.0))

pred = tf.nn.softmax(tf.matmul(x,w)+b)

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels = y)
cost = tf.reduce_mean(entropy+BETA*tf.nn.l2_loss(w)+BETA*tf.nn.l2_loss(b))

global_step = tf.Variable(0, trainable=False, name='g_step')
learning_rate = tf.train.exponential_decay(START_LRATE, global_step,1000, DECAY_RATE, staircase=True)


optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

init = tf.global_variables_initializer()

NO_BATCHES = int(len(train_mat)/BATCH_SIZE)
print("no of batches: ", NO_BATCHES)

train_start = time.time()
with tf.Session() as sess:
    sess.run(init)
    print("started training")
    batches = batchGen(train_mat, train_lab_onehot)
    i=0
    loss = []
    sum_batch_loss = 0
    for batch in batches:
        #avg_cost = 0
        #print("iteration: ", i)
        i+=1
        out,c, step = sess.run([optimizer, cost, global_step], feed_dict={x:batch[0], y:batch[1]})
        sum_batch_loss += c
            
        if i % NO_BATCHES ==0:
            loss.append(sum_batch_loss)
            sum_batch_loss = 0
            print("epoch no:", i/NO_BATCHES, "batch loss=", c, " average loss:", c/BATCH_SIZE, " and global_step= ", step)
            #correct_pred = tf.argmax(pred,1)  
            #acc_dev = 0
            #out_dev = correct_pred.eval({x:dev_mat})
            #for k in range(len(dev_lab_onehot)):
            #    if dev_lab_onehot[k][out_dev[k]] > 0:
            #        acc_dev += 1
            #print("Development Set Accuracy:    ", acc_dev*100/len(dev_lab_onehot))
    
    print("optimization finished")
    train_end = time.time()
    loss_arr = np.array(loss)
    #np.save("/home/ravi/Documents/MLLDS/Assignment2/savedMatrix/losses/simple/exp_dec_lr"+str(int(START_LRATE*100))+"_dec"+str(int(DECAY_RATE*100))+".npy", loss_arr)
    #np.save("/home/ravi/Documents/MLLDS/Assignment2/savedMatrix/losses/simple/exp_inr_lr"+str(int(START_LRATE*100))+"_inr"+str(int(INR_RATE*100))+".npy", loss_arr)
    np.save("/home/dravishankar7/mllds/assign2/savedMatrix/simple/cexp_dec_lr"+str(int(START_LRATE*100))+"_dec"+str(int(DECAY_RATE*100))+".npy", loss_arr)
    #np.save("/home/dravishankar7/mllds/assign2/savedMatrix/simple/cexp_inr_lr"+str(int(START_LRATE*100))+"_inr"+str(int(INR_RATE*100))+".npy", loss_arr)
    
    #correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    #print("Train Accuracy: ", accuracy.eval({x:train_mat, y:train_lab_onehot}))
    correct_pred = tf.argmax(pred,1)    
    traintest_start = time.time()
    acc_train=0
    out_train = correct_pred.eval({x:train_mat})
    for k in range(len(train_lab_onehot)):
        if train_lab_onehot[k][out_train[k]] > 0:
            acc_train += 1
            
    print("Training set accuracy: ", acc_train*100/len(train_lab_onehot))
    del train_lab_onehot, train_mat
    traintest_end = time.time()
    
    dev_mat = np.load("/home/dravishankar7/mllds/assign2/savedMatrix/f_dev_mat.npy")
    dev_lab_onehot = np.load("/home/dravishankar7/mllds/assign2/savedMatrix/f_dev_lab_onehot.npy")
    #dev_mat = np.load("/home/ravi/Documents/MLLDS/Assignment2/savedMatrix/dev_mat.npy")
    #dev_lab_onehot = np.load("/home/ravi/Documents/MLLDS/Assignment2/savedMatrix/dev_lab_onehot.npy")
   
    #print("Development Set Accuracy: ", accuracy.eval({x:dev_mat}))
    acc_dev = 0
    out_dev = correct_pred.eval({x:dev_mat})
    for k in range(len(dev_lab_onehot)):
        if dev_lab_onehot[k][out_dev[k]] > 0:
            acc_dev += 1
    print("Development Set Accuracy: ", acc_dev*100/len(dev_lab_onehot))
    del dev_lab_onehot, dev_mat
    
    test_start = time.time()
    test_mat = np.load("/home/dravishankar7/mllds/assign2/savedMatrix/f_test_mat.npy")
    test_lab_onehot = np.load("/home/dravishankar7/mllds/assign2/savedMatrix/f_test_lab_onehot.npy")
    #test_mat = np.load("/home/ravi/Documents/MLLDS/Assignment2/savedMatrix/test_mat.npy")
    #test_lab_onehot = np.load("/home/ravi/Documents/MLLDS/Assignment2/savedMatrix/test_lab_onehot.npy")
    
    acc_test=0
    out_test = correct_pred.eval({x:test_mat})
    for k in range(len(test_lab_onehot)):
        if test_lab_onehot[k][out_test[k]] > 0:
            acc_test += 1
            
    print("Test set accuracy: ", acc_test*100/len(test_lab_onehot))
    del test_mat, test_lab_onehot
    test_end = time.time()
    
    print("exp learning rate: ", START_LRATE, " decay rate: ", DECAY_RATE)
    #print("exp learning rate: ", START_LRATE, " INR rate: ", INR_RATE)
    print("training time: ", train_end-train_start)
    print("time for training testing: ", traintest_end-traintest_start)
    print("testintg time: ", test_end-test_start)












