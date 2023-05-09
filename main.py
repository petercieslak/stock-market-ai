# ---- IMPORTS ----
import inline as inline
import matplotlib as matplotlib
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
# ------------------

np.set_printoptions(suppress=True)

startDate = datetime(2019, 1, 1)
endDate = datetime(2019, 10, 10)

# Fetching the data
StockDataCsv = pd.read_csv('s&p/sp500_index.csv', skiprows=[0], header=None, names=['Date', 'Closing'])
print(StockDataCsv.shape)
StockDataCsv.head()

FullData = StockDataCsv[['Closing']].values
FullData = FullData.astype('float32')
print(FullData[1:5])
# Feature Scaling for fast training of neural networks

sc=MinMaxScaler()

DataScaler = sc.fit(FullData)
X=DataScaler.transform(FullData)
#X=FullData

print('### After Normalization ###')
print(X[0:5])

# split into samples
X_samples = list()
y_samples = list()

NumerOfRows = len(X)
TimeSteps = 10  # next day's Price Prediction is based on last how many past day's prices

# Iterate thru the values to create combinations
for i in range(TimeSteps, NumerOfRows, 1):
    x_sample = X[i - TimeSteps:i]
    y_sample = X[i]
    X_samples.append(x_sample)
    y_samples.append(y_sample)

################################################
# Reshape the Input as a 3D (number of samples, Time Steps, Features)
X_data = np.array(X_samples)
X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], 1)
print('\n#### Input Data shape ####')
print(X_data.shape)

# We do not reshape y as a 3D data  as it is supposed to be a single column only
y_data = np.array(y_samples)
y_data = y_data.reshape(y_data.shape[0], 1)
print('\n#### Output Data shape ####')
print(y_data.shape)

# Choosing the number of testing data records
TestingRecords = 5

# Splitting the data into train and test
X_train = X_data[:-TestingRecords]
X_test = X_data[-TestingRecords:]
y_train = y_data[:-TestingRecords]
y_test = y_data[-TestingRecords:]

############################################

# Printing the shape of training and testing
print('\n#### Training Data shape ####')
print(X_train.shape)
print(y_train.shape)
print('\n#### Testing Data shape ####')
print(X_test.shape)
print(y_test.shape)


# Defining Input shapes for LSTM
TimeSteps=X_train.shape[1]
TotalFeatures=X_train.shape[2]
print("Number of TimeSteps:", TimeSteps)
print("Number of Features:", TotalFeatures)

# Initialising the RNN
regressor = Sequential()

# Adding the First input hidden layer and the LSTM layer
# return_sequences = True, means the output of every time step to be shared with hidden next layer
regressor.add(LSTM(units=10, activation='relu', input_shape=(TimeSteps, TotalFeatures), return_sequences=True))

# Adding the Second Second hidden layer and the LSTM layer
regressor.add(LSTM(units=5, activation='relu', input_shape=(TimeSteps, TotalFeatures), return_sequences=True))

# Adding the Second Third hidden layer and the LSTM layer
regressor.add(LSTM(units=5, activation='relu', return_sequences=False))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

##################################################

# Measuring the time taken by the model to train
StartTime = time.time()

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, batch_size=5, epochs=10)

EndTime = time.time()
print("## Total Time Taken: ", round((EndTime - StartTime) / 60), 'Minutes ##')

predicted_Price = regressor.predict(X_test)
predicted_Price = DataScaler.inverse_transform(predicted_Price)

print('Predicted price: ', predicted_Price)