import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from Model1_utils import *
from Model2_utils import *
c =10
X_train,X_dev,X_test,Y_train,Y_dev,Y_test,Y = Load_Datasets()
X_train_norm,_,_ =Normalized_data(X_train)
Y_norm, _,_ = Normalized_data(Y[0:100].reshape(1,100))




temp = []
humid = []
Dengue = []
temp_humid = []
wind = []
pop = []
for i in range(X_train.shape[1]):
    temp.append(X_train_norm[0,i])
    humid.append(X_train_norm[3,i])
    wind.append(X_train_norm[1,i])
    Dengue.append(Y[i])
    temp_humid.append(X_train_norm[-1,i])
    pop.append(X_train_norm[-2,i])



X = np.stack((humid,Dengue),axis = 0)


# plt.scatter(X[0,0:50],Y_norm[:,0:50])
#
# plt.title('Pearson relation between Humidity and dengue patients')
# plt.show()

X_mean = X.mean(axis=1, keepdims=True)

X_tild = X - X_mean
X_cor = np.mean(X_tild[0,:]*X_tild[1,:])
X_std = (np.mean(abs(X_tild) ** 2, axis=1, keepdims=True))**0.5
X_pearson_humid = (X_cor/(X_std[0]*X_std[1]))*c



X = np.stack((temp,Dengue),axis = 0)


X_mean = X.mean(axis=1, keepdims=True)

X_tild = X - X_mean
X_cor = np.mean(X_tild[0,:]*X_tild[1,:])
X_std = (np.mean(abs(X_tild) ** 2, axis=1, keepdims=True))**0.5

X_pearson_temp = (X_cor/(X_std[0]*X_std[1]))*c
X = np.stack((temp_humid,Dengue),axis = 0)

# X_cor = np.cov(X)
# X_cor = X_cor[0,0]*X_cor[1,1]
# print(X_cor)
X_mean = X.mean(axis=1, keepdims=True)

X_tild = X - X_mean
X_cor = np.mean(X_tild[0,:]*X_tild[1,:])
X_std = (np.mean(abs(X_tild) ** 2, axis=1, keepdims=True))**0.5

X_pearson_temp_humid = (X_cor/(X_std[0]*X_std[1]))*c
X = np.stack((wind,Dengue),axis = 0)

# X_cor = np.cov(X)
# X_cor = X_cor[0,0]*X_cor[1,1]
# print(X_cor)
X_mean = X.mean(axis=1, keepdims=True)

X_tild = X - X_mean
X_cor = np.mean(X_tild[0,:]*X_tild[1,:])
X_std = (np.mean(abs(X_tild) ** 2, axis=1, keepdims=True))**0.5

X_pearson_wind = (X_cor/(X_std[0]*X_std[1]))*c
X = np.stack((pop,Dengue),axis = 0)

# X_cor = np.cov(X)
# X_cor = X_cor[0,0]*X_cor[1,1]
# print(X_cor)
X_mean = X.mean(axis=1, keepdims=True)

X_tild = X - X_mean
X_cor = np.mean(X_tild[0,:]*X_tild[1,:])
X_std = (np.mean(abs(X_tild) ** 2, axis=1, keepdims=True))**0.5

X_pearson_pop = (X_cor/(X_std[0]*X_std[1]))


# print(X_pearson_humid)
# print(X_pearson_temp)
# print(X_pearson_temp_humid)
# print(X_pearson_wind)
# print(X_pearson_pop)
X=[X_pearson_temp_humid,X_pearson_temp,X_pearson_humid,X_pearson_wind,X_pearson_pop]
print(np.squeeze(X))














# X_cor = np.sum((temp - np.mean(temp))*( Dengue -  np.mean(Dengue)))
# print(X_cor)

# X = np.concatenate((temp,Dengue),axis = 0)


# _,_,sigma1 = Normalized_data(temp)
# _,_,sigma2 = Normalized_data(humid)
# X_cor = np.cov(X)

# print(sigma1)
# print(sigma2)
# print(sigma1.shape)
# print(sigma2.shape)


