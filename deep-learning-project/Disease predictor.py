import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy


import pickle
filename = 'model.sav'
from Model1_utils import *
from Model2_utils import*
X_train,Y_train,X_dev,X_test,Y_dev,Y_test = Load_dataset()
print(X_train.shape)
print(Y_test.shape)
# X_norm = Normalize_data(X_test)
# X_red,S = Visualize_Data(X_train,Y_train)
print(X_red.shape)
#

for i in range(0,100):
    load_params = pickle.load(open(filename, 'rb'))
    Result = Predict_model1(X_test[:, i].reshape(17, 1),X_train, load_params)


    if Result == 0:
        print("Your Blood report is Normal")
    if Result == 1:
        print("You have Symptoms of Dengue according to your blood report ")
    if Result == 2:
        print("you have symptoms of Swine Flu according to your blood report")
    if Result == 3:
        print("you have symptoms of Anemia according to your blood report")
    if Result == 4:
        print(
            "You Don't have any disease among these three  but your report is not Normal,I recommend you to  Consult to your nearest doctor for further checkups ")

#
# X_train, X_dev, X_test, Y_train, Y_dev, Y_test, Y = Load_Datasets()



# plt.xcorr(X_train[0,0:50],Y[0:50],maxlags=9,usevlines=True)
# plt.show()
# scipy.stats.pearsonr(X_train[0,0:50],Y[0:50])