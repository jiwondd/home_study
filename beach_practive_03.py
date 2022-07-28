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

path='./_beach/'
weather = pd.read_csv(path + '해운대 날씨.csv')
customer = pd.read_csv(path + '해운대 입장객수.csv')

print(weather.head(5))
print(weather.shape) #(1618, 11)

# 날짜를 date type으로 변경 후, 나머지는 numeric type으로 변경
weather['날짜'] = pd.to_datetime(weather['날짜'], infer_datetime_format=True)
weather.iloc[:,1:] = weather.iloc[:,1:].apply(pd.to_numeric)

customer['방문일'] = pd.to_datetime(customer['방문일'], infer_datetime_format=True)
customer['방문객수(명)'] = customer['방문객수(명)'].str.replace(",","")
customer.iloc[:,1:] = customer.iloc[:,1:].apply(pd.to_numeric)

# weather data filtering
weather = weather[(weather['날짜'] >= '2021-06-01') & (weather['날짜'] <= '2021-08-31')].reset_index(drop=True)

# customer data drop_duplicates
customer = customer.drop_duplicates().reset_index(drop=True)

# merge data
total_data = pd.merge(weather, customer, left_on='날짜', right_on="방문일", how='inner')
total_data = total_data[['강수_관측값', "기온", "습도", "체감온도", "평균 수온", "평균 풍속", "평균기압", "평균최대파고", "방문객수(명)"]]

# train/test split
train_x, test_x, train_y, test_y = train_test_split(total_data.iloc[:, :-1], total_data['방문객수(명)'], test_size=0.2)

# minmax scaler
x_mm_scaler = MinMaxScaler()
train_x_scaled = x_mm_scaler.fit_transform(train_x)
test_x_scaled = x_mm_scaler.transform(test_x)

print(test_x_scaled.shape) #(19, 8)

model=Sequential()
model.add(Dense(32,input_dim=8))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))
# model.summary()
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 32)                288
# _________________________________________________________________
# dense_1 (Dense)              (None, 64)                2112
# _________________________________________________________________
# dense_2 (Dense)              (None, 128)               8320
# _________________________________________________________________
# dense_3 (Dense)              (None, 64)                8256
# _________________________________________________________________
# dense_4 (Dense)              (None, 32)                2080
# _________________________________________________________________
# dense_5 (Dense)              (None, 16)                528
# _________________________________________________________________
# dense_6 (Dense)              (None, 1)                 17
# =================================================================
# Total params: 21,601
# Trainable params: 21,601
# Non-trainable params: 0
# _________________________________________________________________

model.compile(loss='mse',optimizer='adam')
model.fit(train_x_scaled,train_y,epochs=50,batch_size=16,validation_split=0.2)

loss=model.evaluate(test_x,test_y)
pred_y=model.predict(test_x)

print('loss: ',loss)
print('예상 입장객 수: ', pred_y[-1:])
# print('예상 입장객 수: ', pred_y)
# loss:  14157799424.0
# 예상 입장객 수:  [[168358.5 ]
#  [182817.61]
#  [206973.45]
#  [163275.95]
#  [165539.64]
#  [166794.11]
#  [164681.83]
#  [164396.52]
#  [190933.14]
#  [165763.08]
#  [162650.8 ]
#  [166222.92]
#  [165761.98]
#  [181281.23]
#  [171040.03]
#  [167051.89]
#  [156354.36]
#  [152092.  ]
#  [161799.11]]

# loss:  155227432943616.0
# 예상 입장객 수:  [[12455927.]]

