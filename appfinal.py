#Import Library

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
import streamlit as st

#Initialize range of Date
start='2015-04-30'
end='2023-05-01'

st.title('Intelligent Stock Market Prediction')

user = st.text_input('Enter Stock Ticker:', 'MCD')     #User Stock Market Input
stock = yf.download(user, start, end)                   #Extracting Data from Yahoo Finance

st.subheader('Data from 2015 to 2023')
st.write(stock.describe())

#Closing Price Plot
st.subheader('Closing Price vs Time')
fig = plt.figure(figsize=(12,6))
plt.plot(stock.Close)
plt.xlabel('Time (Years)')
plt.ylabel('Price (USD)')
st.pyplot(fig)

#Moving Average Plot
ma100 = stock.Close.rolling(100).mean()
ma200 = stock.Close.rolling(200).mean()

st.subheader('Closing Price with Moving Average 100 & 200 indicators')
fig2 = plt.figure(figsize = (12, 6))
plt.plot(stock.Close)
plt.plot(ma100)
plt.plot(ma200)
plt.xlabel('Time (Years)')
plt.ylabel('Price (USD)')
plt.legend(['Close Price', 'ma100', 'ma200'], loc='lower right')
st.pyplot(fig2)

#Data splitting into Training and Testing with 80:20 ratio
datalen = int(len(stock))
trainlen = int(len(stock)*0.80)

train = pd.DataFrame(stock['Close'][0 : trainlen])
test = pd.DataFrame(stock['Close'][trainlen : datalen])

#Data Preprocessing (Normalize and rescale data)
scaler = MinMaxScaler(feature_range=(0,1))
train_array = scaler.fit_transform(train)

#Loading of previously trained Model
model = load_model('StockPredict_model.h5')

#Data Testing
data = stock.filter(['Close'])
values = data.values
scaled_data = scaler.fit_transform(values)
testarray = scaled_data[trainlen-100: , : ]

xtest = []
ytest = data[trainlen : ]

for i in range(100, len(testarray)):
    xtest.append(testarray[i-100 : i, 0])

xtest = np.array(xtest)
xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))

#Data Prediction, Calculation of Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) 
ypredict = model.predict(xtest)
ypredict = scaler.inverse_transform(ypredict)

mse = mean_squared_error(ytest, ypredict)
rmse = np.sqrt(mse)

mae = mean_absolute_error(ytest, ypredict)

#Prediction vs Actual Closing Price Plot
st.subheader('Prediction Price vs Actual Price')
fig3 = plt.figure(figsize=(12,6))
validation = data[trainlen : ]
validation['ypredict'] = ypredict
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price ($)')
plt.plot(train['Close'])
plt.plot(validation[['Close', 'ypredict']])
plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')
st.pyplot(fig3)

#RMSE and MAE
st.subheader('Root Mean Square Error and Mean Absolute Error')
st.write('RMSE = ', rmse)
st.write('MAE = ', mae)
