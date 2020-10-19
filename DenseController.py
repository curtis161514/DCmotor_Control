# -*- coding: utf-8 -*-

#Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibility

import tensorflow as tf
import pandas as pd 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Convolution2D, MaxPooling2D
#from tensorflow.keras.utils import np_utils
from tensorflow.keras import optimizers


#load csv
filename = 'DCMotorData.csv' 
data = pd.read_csv(filename)  
data.describe()


#Create test set dataframe and devide by 100 to normalize controlvalve positions
test_data = data[['Voltage']].copy()
datant = (test_data -test_data.min())/(test_data.max()-test_data.min())
print(datant)
list(test_data.columns.values)

#create a train set data
train_data = data.drop(['Voltage'], axis=1)
list(train_data.columns.values)

# Normalize train data
datan = (train_data - train_data.min()) / (train_data.max() - train_data.min())
print(datan)

#replace Nan values with 0
#datan["LIC1110_SV"] = 0
#print (datan)

#check data types and then change to float 32
datan = datan.astype('float32')
datant = datant.astype('float32')
#print(datan.dtypes)

ncolx = datan.shape[1] #number of columns in data train X
ncoly = datant.shape[1] #number of columns in data train y
nrows = datan.shape[0] #number of rows of training data

#calculate seperate 60/30/10 train test val splits
trainset = int(round(.6*nrows,0))
testset = int(round(.3*nrows,0))
valset = trainset + testset


#Shuffle the Data and create train test dataframes
perm = np.random.permutation(nrows)


x_train = datan.iloc[perm[0:trainset], 0:ncolx]
print(x_train)

y_train = datant.iloc[perm[0:trainset], 0:ncoly]
print(y_train)

x_test  = datan.iloc[perm[trainset:valset], 0:ncolx]
#print(x_test)

y_test = datant.iloc[perm[trainset:valset], 0:ncoly]
#print(y_test)

x_val  = datan.iloc[perm[valset:nrows], 0:ncolx]
#print(x_val)

y_val = datant.iloc[perm[valset:nrows], 0:ncoly]
#print(y_val)


#convert to np array
X_train= np.asarray(x_train)
Y_train= np.asarray(y_train)
X_test= np.asarray(x_test)
Y_test= np.asarray(y_test)
X_val= np.asarray(x_val)
Y_val= np.asarray(y_val)


print(X_train)
print(Y_train)


#clear previous model
tf.keras.backend.clear_session()

#Define model architecture
model = Sequential()#Define model architecture
model = Sequential()

#design nn arcitecture
model.add(Dense(14, input_shape = (ncolx,), activation = 'relu'))
model.add(Dense(14, activation = 'relu'))
model.add(Dense(14, activation = 'relu'))
model.add(Dense(ncoly, activation = 'linear'))
 
# Compile model
model.compile(loss='mse', optimizer= 'rmsprop')
model.summary()

# Fit model on training data
model.fit(X_train, Y_train, 
          batch_size=200, epochs=1000, verbose=1, validation_data = (X_test,Y_test))
 


# Fit model on training data
model.fit(X_train, Y_train, 
          batch_size=200, epochs=1000, verbose=1, validation_data = (X_test,Y_test))
 

# Evaluate model on test data
score = model.evaluate(X_val, Y_val, verbose=0)
score = round(score*1000,3)
print(score)

print(model.predict(X_val))
print(Y_val)



#create unique string to save model to working directory

d = str(score) + '_Dense.h5'
model.save(d)

#create dataframe of max min mean and stdev values
max_min = {'min': train_data.min(), 'max': train_data.max(),'mean':train_data.mean(),'stdev': train_data.std()}
max_min = pd.DataFrame(data = max_min)

#save dataframe to file
e = str(score) + '_Inputs.txt'
max_min.to_csv(e,sep='\t')

#save outputs to file
max_min1 = {'min': test_data.min(), 'max': test_data.max(),'mean':test_data.mean(),'stdev': test_data.std()}
max_min1 = pd.DataFrame(data = max_min1)

o = str(score) + '_Outputs.txt' 
max_min1.to_csv(o,sep='\t')

