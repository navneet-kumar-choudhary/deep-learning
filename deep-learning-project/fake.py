import numpy as np
from Model1_utils import *
import matplotlib.pyplot as plt
import pickle
from Model2_utils import *
from Model1_utils import *

parameters2 = pickle.load(open('model2_new.sav','rb'))
parameters1 = pickle.load(open('model.sav','rb'))
# print(parameters)
X_train1, Y_train1, X_dev1, X_test1, Y_dev1, Y_test1 =Load_dataset()
# #normalising our data
#
# print(X_train_norm)
X_train,X_dev,X_test,Y_train,Y_dev,Y_test,Y1= Load_Datasets()
print(X_train.shape)
print(X_test.shape)
print(X_train1.shape)
print(X_test1.shape)
# print(X_norm)

# X_train1, Y_train1, X_dev1, X_test1, Y_dev1, Y_test1 =Load_dataset()

# X,Y,X_demo = Kerala_data_Load()

# print(X_demo.shape)

# PCA(X_train,4)







# #
# Output2 = Predict_model2(X_train,X_train,parameters2)
# Output1 = Predict_model1(X_test[:,2],X_train,parameters1)
# print(Output2.shape)
# print(Output2)


# accuracy  = tf.reduce_mean(tf.cast(tf.equal(Output2,tf.argmax(Y_train)),'float'))*100
# print(tf.Session().run(accuracy))
# Output1 = Predict_model1(X_train1,X_train1,parameters1)
# accuracy  = tf.reduce_mean(tf.cast(tf.equal(Output1,tf.argmax(Y_train1)),'float'))*100
# print(tf.Session().run(accuracy))

# Hist(parameters2)#

# print(X_train_norm)
# print(X_train_norm1)
# PCA(X_train_norm1,5)
# X_train_norm1 = Normalized_data(X_train)
# Co_relation(i = 'Humidity')
# Co_relation(i = 'Temperature')
# Co_relation(i ='Both')
# print()


