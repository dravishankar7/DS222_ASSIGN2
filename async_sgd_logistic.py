#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 00:09:10 2017

@author: ravi
"""


import time
import numpy as np
import tensorflow as tf
import random
import threading


MAX_WORDS = 10000
#NO_LABLES = 49
L_RATE = 5
TRAINING_EPOCHS = 5
BATCH_SIZE = 64
BETA = 0.001
NUM_CONCURRENT_STEPS = 4




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


X_train = train_mat
y_train = train_lab_onehot

NO_LABLES = len(train_lab_onehot[0])

print("loaded data")

def batchGen(train_mat, train_lab_onehot, seed):
    random.seed(seed)
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

optimizer = tf.train.GradientDescentOptimizer(learning_rate=L_RATE).minimize(cost)

init = tf.global_variables_initializer()

NO_BATCHES = int(len(train_mat)/BATCH_SIZE)
print("no of batches: ", NO_BATCHES)

train_start = time.time()
with tf.Session() as sess:
    sess.run(init)
    print("started training")
    def ASG(proc_id):
        seed = proc_id*54321
        batches = batchGen(X_train,y_train,seed)
        i=0
        loss = []
        sum_batch_loss = 0
        for batch in batches:
            i+=1
            out,c = sess.run([optimizer, cost], feed_dict={x:batch[0], y:batch[1]})
            sum_batch_loss += c
            
            if i % NO_BATCHES ==0:
                loss.append(sum_batch_loss)
                sum_batch_loss = 0
                print("epoch no:", i/NO_BATCHES, "batch loss=", c, " average loss:", c/BATCH_SIZE)
        np.save('/home/dravishankar7/mllds/assign2/savedMatrix/ASG/asg_loss_constLR'+str(L_RATE*100)+"_id"+str(proc_id) +'.npy',np.array(loss))
        #np.save('/home/ravi/Documents/MLLDS/Assignment2/savedMatrix/losses/ASG/asg_loss_constLR'+str(L_RATE*100)+"_id"+str(proc_id) +'.npy',np.array(loss))  
    
    
    train_threads = []
    for _ in range(NUM_CONCURRENT_STEPS):
        train_threads.append(threading.Thread(target=ASG, args=(_,)))

    for t in train_threads:
       t.start()
    for t in train_threads:
       t.join()

    
    print("optimization finished")
    train_end = time.time()
    
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
    
    print("Learning rate: ", L_RATE)
    print("no.of epochs: ", TRAINING_EPOCHS)
    print("training time: ", train_end-train_start)
    print("time for training testing: ", traintest_end-traintest_start)
    print("testintg time: ", test_end-test_start)












