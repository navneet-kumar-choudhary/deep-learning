import numpy as np
import pandas as pd
import tensorflow as tf
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import colors
import tensorflow as tf
def Load_Datasets():
   df1 = pd.read_csv("DataSet/total_mapped_weather_2.csv")
   df2 = pd.read_csv('DataSet/total_mapped_weather_dengue.csv')


   Y = np.array(df2).T

   X = np.array(df1).T
   X_humid = X[7,:]

   X_temp = X[4,:]

   X_rat = X_temp/X_humid

   X_train = X[4:8,0:100000]
   X_train = np.concatenate((X_train,X[2,0:100000].reshape((1,100000))),axis=0)
   X_train = np.concatenate((X_train,X_rat[0:100000].reshape(1,100000)),axis = 0)
   X_dev = X[4:8,200000:255984]
   X_dev = np.concatenate((X_dev,X[2,200000:255984].reshape((1,55984))),axis=0)
   X_dev = np.concatenate((X_dev,X_rat[200000:255984].reshape(1,55984)),axis = 0)
   X_test = X[4:8,255984:]
   X_test = np.concatenate((X_test,X[2,255984:].reshape((1,55984))),axis=0)
   X_test = np.concatenate((X_test,X_rat[255984:].reshape(1,55984)),axis = 0)
   Y = Y[-1, :]
   Y1 = np.zeros((Y.shape[0], 5)).T
   for i in range(Y.shape[0]):
      if Y[i] <= 5:
         rate = 0
         Y1[rate, i] = 1
      if Y[i] <= 30 and Y[i] > 5:
         rate = 1
         Y1[rate, i] = 1
      if Y[i] <= 50 and Y[i] > 30:
         rate = 2
         Y1[rate, i] = 1
      if Y[i] <= 80 and Y[i] > 50:
         rate = 3
         Y1[rate, i] = 1
      if Y[i] > 80:
         rate = 4
         Y1[rate, i] = 1
   Y_train = Y1[:,0:100000]
   Y_dev = Y1[:,200000:255984]
   Y_test  = Y1[:,255984:]
   X_train.astype(float)
   X_dev.astype(float)
   X_test.astype(float)

   return X_train,X_dev,X_test,Y_train,Y_dev,Y_test,Y
def Kerala_data_Load():
    df3 = pd.read_csv("Dataset/kerela_mapped_weather_dengue.csv")
    X = np.array(df3).T
    m = X.shape[1]
    X_train = X[4:8,:]
    X_humid = X[7, :]


    X_temp = X[4, :]


    X_rat = X_temp / X_humid
    X_train = np.concatenate((X_train,X[2,:].reshape((1,m))),axis = 0)
    X_train = np.concatenate((X_train, X_rat.reshape(1, m)), axis=0)
    Y = X[-2,:]
    # df4 = pd.read_csv('Dataset/total_mapped_weather_dengue_2.csv')
    # X_demo = np.array(df4).T
    #
    Y_train = np.zeros((5, Y.shape[0]))
    for i in range(Y.shape[0]):
        if Y[i] <= 5:
            rate = 0
            Y_train[rate, i] = 1
        if Y[i] <= 30 and Y[i] > 5:
            rate = 1
            Y_train[rate, i] = 1
        if Y[i] <= 50 and Y[i] > 30:
            rate = 2
            Y_train[rate, i] = 1
        if Y[i] <= 80 and Y[i] > 50:
            rate = 3
            Y_train[rate, i] = 1
        if Y[i] > 80:
            rate = 4
            Y_train[rate, i] = 1
    return X_train,Y_train

def Normalized_data(X):
   X_mean = X.mean(axis=1, keepdims=True)

   X_tild = X - X_mean
   X_std = (np.mean(abs(X_tild) ** 2, axis=1, keepdims=True)) ** 0.5
   X_norm = X_tild / X_std
   return X_norm,X_mean,X_std
def Co_relation(i):
   X_train,_,_,_,_,_,Y = Load_Datasets()
   if i == 'Humidity':
      fig, ax1 = plt.subplots()

      line1 = ax1.plot(X_train[-4, 0:100], c='r', label='mean Relative Humidity in %')
      ax1.set_xlabel('Daywise Variations')
      ax1.set_ylabel('MHR ', color="red")
      ax1.tick_params(axis='y', labelcolor='red')

      ax2 = ax1.twinx()
      ax2.set_ylabel('No. of dengue Patients', color='green')
      line3 = ax2.plot(Y[0:100], c='g', label='No. of Dengue Patients')
      ax2.tick_params(axis='y', labelcolor='green')

      ax1.legend()
      ax2.legend()
      fig.tight_layout()

      plt.show()

   if i == 'Temperature':
      fig,ax1 = plt.subplots()
      line2 = ax1.plot(X_train[0, 0:100], c='b', label='mean Temp. in Celcius')
      ax1.tick_params(axis='y', labelcolor='blue')
      ax1.set_xlabel('Daywise Variations')
      ax1.set_ylabel('MTR ', color="blue")
      ax2 = ax1.twinx()

      line3 = ax2.plot(Y[0:100], c='g', label='No. of Dengue Patients')
      ax2.set_ylabel('No. of dengue Patients', color='green')
      ax2.tick_params(axis='y', labelcolor='green')

      ax1.legend()
      ax2.legend()
      fig.tight_layout()


      plt.show()
   if i== 'Both':
      fig, ax1 = plt.subplots()

      line1 = ax1.plot(X_train[-4, 0:100], c='r', label='mean Relative Humidity in %')
      ax1.set_xlabel('Daywise Variations')
      ax1.set_ylabel('MHR & MTR', color="red")
      line2 = ax1.plot(X_train[0, 0:100], c='b', label='mean Temp. in Celcius')


      ax1.tick_params(axis='y', labelcolor='red')

      ax2 = ax1.twinx()
      line3 = ax2.plot(Y[0:100], c='g', label='No. of Dengue Patients')
      ax2.set_ylabel('No. of dengue Patients', color='green')
      ax2.tick_params(axis='y', labelcolor='green')

      ax1.legend()
      ax2.legend()
      fig.tight_layout()

      plt.show()




def compute_cost2(Z5, Y, parameters, beta):
      W1 = parameters['W1']
      b1 = parameters['b1']
      W2 = parameters['W2']
      b2 = parameters['b2']
      W3 = parameters['W3']
      b3 = parameters['b3']
      W4 = parameters['W4']
      b4 = parameters['b4']
      W5 = parameters['W5']
      b5 = parameters['b5']
      logits = tf.transpose(Z5)
      labels = tf.transpose(Y)
      cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
      regularizer = beta * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(W5))
      cost = tf.reduce_mean(cost + regularizer)
      return cost
def feed_forward2(X,parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']
    Z1= tf.add(tf.matmul(W1,X),b1)
    A1=tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)
    A3 = tf.nn.relu(Z3)
    Z4 = tf.add(tf.matmul(W4,A3),b4)
    A4 = tf.nn.relu(Z4)
    Z5 = tf.add(tf.matmul(W5,A4),b5)
    return Z5
def initialize_parameters2(n_l,nodes,n):
    W1 = tf.get_variable("W1", [nodes[0], n],dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [nodes[0], 1], dtype=tf.float64,initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [nodes[1], nodes[0]],dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [nodes[1], 1],dtype=tf.float64, initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [nodes[2], nodes[1]],dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [nodes[2], 1], dtype=tf.float64,initializer=tf.zeros_initializer())
    W4 = tf.get_variable("W4", [nodes[3], nodes[2]],dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", [nodes[3], 1],dtype=tf.float64, initializer=tf.zeros_initializer())
    W5 = tf.get_variable('W5',[nodes[4],nodes[3]],dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.get_variable('b5',[nodes[4],1],dtype=tf.float64, initializer=tf.zeros_initializer())
    parameters = {'W1': W1,
                  'W2':W2,
                  'W3':W3,
                  'W4':W4,
                  'b1':b1,
                  'b2':b2,
                  'b3':b3,
                  'b4':b4,
                   'W5':W5,
                  'b5':b5}
    return parameters
def PCA(X,k):
   X,_,_= Normalized_data(X)
   X_co = np.dot(X,X.T)
   U,S,_ = np.linalg.svd(X_co,full_matrices=True)
   print(np.sum(S[0:k])/np.sum(S))
   # x = tf.placeholder(dtype=tf.float64)
   # s, u, v = tf.linalg.svd(x)
   # tf_a_approx = tf.matmul(u, tf.matmul(tf.linalg.diag(s), v, adjoint_b=True))
   # u, s, v_adj = np.linalg.svd(x, full_matrices=False)
   # np_a_approx = np.dot(u, np.dot(np.diag(s), v_adj))











def Hist(parameters):
    X_train, X_dev, X_test, Y_train, Y_dev, Y_test, Y = Load_Datasets()
    Output2 = Predict_model2(X_train,X_train,parameters)
    Actual = tf.Session().run(tf.argmax(Y_train))
    plt.hist(Actual[0:100], alpha=0.5, label='Actual Data ')
    plt.hist(Output2[0:100],alpha = 0.5,label = 'Model prediction')

    plt.legend(loc='upper right')
    plt.xlabel('Degree of Dengue')
    plt.ylabel('No. of times  dengue of a particular degree  occured ')
    plt.title('Model Accuracy')



    plt.show()
def Predict_model2(X,X_train,parameters):
    _,X_mean,X_std = Normalized_data(X_train)
    X = (X-X_mean)/X_std

    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    W4 = tf.convert_to_tensor(parameters['W4'])
    b4 = tf.convert_to_tensor(parameters['b4'])
    W5 = tf.convert_to_tensor(parameters['W5'])
    b5 = tf.convert_to_tensor(parameters['b5'])

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3,
              'W4':W4,
              'b4':b4,
              'W5':W5,
              'b5':b5}
    x =tf.placeholder(dtype=tf.float64)
    Z = feed_forward2(x,params)
    p = tf.argmax(Z)
    with tf.Session() as sess:
        Output = sess.run(p,feed_dict={x:X})
    return Output








