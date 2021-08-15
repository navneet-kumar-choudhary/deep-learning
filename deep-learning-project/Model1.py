import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops


from Model1_utils import*
import pickle



#loading the parameters
X_train, Y_train, X_dev, X_test, Y_dev, Y_test =Load_dataset()
#normalising our data
X_train_norm,_,_ = Normalize_data(X_train)
X_dev_norm,_,_ = Normalize_data(X_dev)
X_test_norm,_,_ = Normalize_data(X_test)
# X_red = Visualize_Data(X_train,Y_train)

C = 5
n = X_train.shape[0]
m = X_train.shape[1]
# no. of layers
nodes=[50,20,10,C]              #list containing no. of nodes in each layer
n_l = len(nodes)
costs = []
num_epochs = 1200
learning_rate = 0.0001
mini_batch_size = 32
beta = 0.01

def model(X_train,Y_train,X_test,Y_test,X_dev,Y_dev,num_epochs,learning_rate,mini_batch_size,n_l,nodes,beta):
    ops.reset_default_graph()
    X = tf.placeholder(dtype=tf.float64)
    Y = tf.placeholder(dtype=tf.int64)

    parameters = initialize_parameters(n_l,nodes,n)

    Z = feed_forward(X,parameters)

    cost = compute_cost(Z,Y,parameters,beta)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for epochs in range(num_epochs):


            epoch_cost=0
            mini_batches = random_miniBatches(X_train_norm, Y_train, mini_batch_size)
            num_miniBatch = int(m/mini_batch_size)
            for mini_Batch in mini_batches:
                (miniBatch_X, miniBatch_Y) = mini_Batch
                _,mini_batch_cost=sess.run([optimizer,cost],feed_dict={X:miniBatch_X,Y:miniBatch_Y})
                epoch_cost += mini_batch_cost/num_miniBatch



            if epochs % 100 == 0:
                print('Cost after epoch %i %f' % (epochs, epoch_cost))





            if epochs %20 == 0:
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
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))*100
        print("Train Accuracy:", accuracy.eval({X: X_train_norm, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test_norm, Y: Y_test}))
        print("Dev Accuracy:", accuracy.eval({X: X_dev_norm, Y: Y_dev}))

    return parameters



parameters = model(X_train,Y_train,X_test,Y_test,X_dev,Y_dev,num_epochs,learning_rate,mini_batch_size,n_l,nodes,beta)
filename = 'model.sav'
pickle.dump(parameters,open(filename,'wb'))


