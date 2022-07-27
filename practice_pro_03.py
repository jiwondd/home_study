import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.layers import Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import time
from sklearn.model_selection import train_test_split
import inspect, os

path='./_data/'
weather = pd.read_csv(path + '해운대 날씨.csv')
customer = pd.read_csv(path + '해운대 입장객수.csv')

print(weather.head(5))

# 날짜를 date type으로 변경 후, 나머지는 numeric type으로 변경
weather['날짜'] = pd.to_datetime(weather['날짜'], infer_datetime_format=True)
weather.iloc[:,1:] = weather.iloc[:,1:].apply(pd.to_numeric)

customer['방문일'] = pd.to_datetime(customer['방문일'], infer_datetime_format=True)
customer['방문객수(명)'] = customer['방문객수(명)'].str.replace(",","")
customer.iloc[:,1:] = customer.iloc[:,1:].apply(pd.to_numeric)

print(weather.info())

# weather data filtering
weather = weather[(weather['날짜'] >= '2021-06-01') & (weather['날짜'] <= '2021-08-31')].reset_index(drop=True)

# customer data drop_duplicates
customer = customer.drop_duplicates().reset_index(drop=True)

# merge data
total_data = pd.merge(weather, customer, left_on='날짜', right_on="방문일", how='inner')
total_data = total_data[['강수_관측값', "기온", "습도", "체감온도", "수온", "평균풍속", "평균기압", "최대파고", "방문객수(명)"]]

# train/test split
train_x, test_x, train_y, test_y = train_test_split(total_data.iloc[:, :-1], total_data['방문객수(명)'], test_size=0.2)

# minmax scaler
x_mm_scaler = MinMaxScaler()
train_x_scaled = x_mm_scaler.fit_transform(train_x)
test_x_scaled = x_mm_scaler.transform(test_x)

print(test_x_scaled.shape) #(19, 8)

