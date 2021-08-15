import pandas as pd
import numpy as np
import tensorflow as tf
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from Model2_utils import *

import json
def Load_dataset():

    df1 = pd.read_csv(r'DataSet/training_data.csv')
    df2= pd.read_csv("DataSet/training_label.csv")
    df3 = pd.read_csv("DataSet/dev_data.csv")
    df4 = pd.read_csv("DataSet/dev_label.csv")
    df5 = pd.read_csv("DataSet/test_data.csv")
    df6 = pd.read_csv("DataSet/test_label.csv")
    # print(df1)
    X_train = np.array(df1).T
    Y_train=np.array(df2).T
    X_dev = np.array(df3).T
    Y_dev = np.array(df4).T
    X_test = np.array(df5).T
    Y_test = np.array(df6).T

    return X_train,Y_train,X_dev,X_test,Y_dev,Y_test

def Normalize_data(X):

    X_mean = X.mean(axis=1,keepdims=True)
    X_std = X.std(axis=1,keepdims=True)
    X_norm = (X - X_mean) / X_std
    return X_norm,X_mean,X_std
def random_miniBatches(X,Y,mini_batch_size):
    mini_Batches=[]
    m = X.shape[1]
    a = np.random.permutation(m)
    X=X[:,a]
    Y=Y[:,a]
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0,num_complete_minibatches):
        miniBatch_X=X[:,k*mini_batch_size:k*mini_batch_size+mini_batch_size]
        miniBatch_Y=Y[:,k*mini_batch_size:k*mini_batch_size+mini_batch_size]
        miniBatch=(miniBatch_X,miniBatch_Y)
        mini_Batches.append(miniBatch)

    if m % mini_batch_size != 0:
        miniBatch_X = X[:,num_complete_minibatches * mini_batch_size : m]
        miniBatch_Y = Y[:,num_complete_minibatches * mini_batch_size : m]
        miniBatch = (miniBatch_X, miniBatch_Y)
        mini_Batches.append(miniBatch)


    return mini_Batches



def initialize_parameters(n_l,nodes,n):
    W1 = tf.get_variable("W1", [nodes[0], n],dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [nodes[0], 1], dtype=tf.float64,initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [nodes[1], nodes[0]],dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [nodes[1], 1],dtype=tf.float64, initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [nodes[2], nodes[1]],dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [nodes[2], 1], dtype=tf.float64,initializer=tf.zeros_initializer())
    W4 = tf.get_variable("W4", [nodes[3], nodes[2]],dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", [nodes[3], 1],dtype=tf.float64, initializer=tf.zeros_initializer())
    parameters = {'W1': W1,
                  'W2':W2,
                  'W3':W3,
                  'W4':W4,
                  'b1':b1,
                  'b2':b2,
                  'b3':b3,
                  'b4':b4}
    return parameters




def feed_forward(X,parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    Z1= tf.add(tf.matmul(W1,X),b1)
    A1=tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)
    A3 = tf.nn.relu(Z3)
    Z4 = tf.add(tf.matmul(W4,A3),b4)
    return Z4

def compute_cost(Z4,Y,parameters,beta):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    logits=tf.transpose(Z4)
    labels=tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    regularizer = beta*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4) )
    cost = tf.reduce_mean(cost + regularizer)
    return cost


def Predict_model1(X,X_train,parameters):
    _, X_mean, X_std = Normalize_data(X_train)
    X = (X - X_mean) / X_std
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    W4 = tf.convert_to_tensor(parameters['W4'])
    b4 = tf.convert_to_tensor(parameters['b4'])
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3,
              'W4': W4,
              'b4': b4,}
    x = tf.placeholder(dtype=tf.float64)
    Z = feed_forward(x, params)
    p = tf.argmax(Z)
    with tf.Session() as sess:
        Output = sess.run(p, feed_dict={x: X})
    return Output




    return Output
def Visualize_Data(X,Y):

    X,_,_ = Normalize_data(X)
    X_co = np.dot(X , np.transpose(X))
    U,S,_ = np.linalg.svd(X_co,full_matrices=True)
    X_red = np.dot(np.transpose(X),U[:,0:3])
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    line1 = ax.scatter(X_red[0:319,0],X_red[0:319,1],X_red[0:319,2],s=4,c='r',marker='o')
    line2 = ax.scatter(X_red[320:380, 0], X_red[320:380, 1], X_red[320:380, 2], s=4, c='b', marker='o')
    line3 = ax.scatter(X_red[380:440, 0], X_red[380:440, 1], X_red[380:440, 2], s=4, c='g', marker='o')
    line4 = ax.scatter(X_red[440:500, 0], X_red[440:500, 1], X_red[440:500, 2], s=4, c='k', marker='o')

    ax.legend((line1,line2,line3,line4),('Normal','Dengue','Anemia','Swine-flu'))
    plt.title('Visualization of our Data')
    plt.show()



    return X_red,S







