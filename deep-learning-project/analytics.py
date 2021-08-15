import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from Model2_utils import *
from Model1_utils import *
import random
import pickle
x=np.zeros((235,32))

df2 = pd.read_csv(r'DataSet/kerela_mapped_weather_dengue.csv')
df1 = pd.read_csv(r'DataSet/training_data.csv')
df3 = pd.read_csv("DataSet/training_label.csv")
X = np.array(df2).T
m = X.shape[1]
X_train = X[4:8,:]
X_humid = X[7, :]


X_temp = X[4, :]

X_rat = X_temp / X_humid
X_train = np.concatenate((X_train,X[2,:].reshape((1,m))),axis = 0)
X_train = np.concatenate((X_train, X_rat.reshape(1, m)), axis=0)
Y = X[-2,:]
lat = X[0,0:m:32]
print(lat.shape)
lng = X[1,0:m:32]

parameters = pickle.load(open('model2_new.sav','rb'))
print(parameters['W1'].shape)

steps =32#int(X_train.shape[1]/235)
Output = Predict_model2(X_train,X_train,parameters)


# for i in range(235):
#     x[i,:] = Output[i*steps:i*steps+steps]
#
# print(x)

print(Output.shape[0])
for i in range(Output.shape[0]):
    if Output[i] == 0:
        Output[i] = random.randint(0,5)
    elif Output[i] == 1:
        Output[i] = random.randint(5,30)
    elif Output[i] == 2:
        Output[i] = random.randint(30,50)
    elif Output[i] == 3:
        Output[i] = random.randint(50,80)
    elif Output[i] == 4:
        Output[i] = random.randint(80,100)


for i in range(235):
    x[i,:] = Output[i*steps:i*steps+steps]


select  = input("enter index of which you want to see variations\n")
select = int(select)
print("selected lat = " + str(lat[select]) + " " + 'selected lang  = '+ str(lng[select]))
plt.plot(x[select,:])
plt.title("Lat" + ':' + str(lat[select]) + "    " + "Lang"+ ':'  + str(lng[select]))
plt.show()









