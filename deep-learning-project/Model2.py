import numpy as np
import pandas as pd
import tensorflow as tf
import math
from tensorflow.python.framework import ops
import pickle
import matplotlib.pyplot as plt
from Model2_utils import *
from Model1_utils import *
X_train,X_dev,X_test,Y_train,Y_dev,Y_test,_ = Load_Datasets()
X_train_kerala, Y_train_kerala,X_demo = Kerala_data_Load()
X_train_kerala_norm,_,_ = Normalized_data(X_train_kerala)
X_train_norm,_,_ = Normalized_data(X_train)
X_dev_norm,_,_ = Normalized_data(X_dev)
X_test_norm,_,_ = Normalized_data(X_test)



m = X_train.shape[1]
n = X_train.shape[0]
C = 5
num_epochs =600
mini_batch_size = 64
beta = 0
costs=[]

print(X_train_norm.shape)



nodes = [50,25,15,10,5]
n_l = len(nodes)
learning_rate = 0.0001
def Model2(X_train,X_dev,X_test,Y_train,Y_dev,Y_test,n_l,learning_rate,num_epochs,mini_batch_size,nodes,beta,n):
    ops.reset_default_graph()
    X = tf.placeholder(dtype=tf.float64)
    Y = tf.placeholder(dtype=tf.int64)
    parameters = initialize_parameters2(n_l, nodes, n)
    Z = feed_forward2(X,parameters)
    cost = compute_cost2(Z,Y,parameters,beta)
    optimizer =  tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epochs in range(num_epochs):
            epoch_cost=0
            mini_batches = random_miniBatches(X_train_norm,Y_train,mini_batch_size)
            num_miniBatch  = int(m/mini_batch_size)
            for mini_batch in  mini_batches:
                (miniBatch_X,miniBatch_Y) = mini_batch
                _, mini_batch_cost = sess.run([optimizer, cost], feed_dict={X: miniBatch_X, Y: miniBatch_Y})
                epoch_cost += mini_batch_cost / num_miniBatch

            if epochs % 100 == 0:
                print('Cost after epoch %i %f' % (epochs, epoch_cost))

            if epochs % 20 == 0:
                costs.append(epoch_cost)

                # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

            # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

            # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))

            # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * 100
        print("Train Accuracy:", accuracy.eval({X: X_train_norm, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test_norm, Y: Y_test}))
        print("Dev Accuracy:", accuracy.eval({X: X_dev_norm, Y: Y_dev}))

    return parameters
parameters = Model2(X_train,X_dev,X_test,Y_train,Y_dev,Y_test,n_l,learning_rate,num_epochs,mini_batch_size,nodes,beta,n)
filename = 'model2_new.sav'
pickle.dump(parameters,open(filename,'wb'))





