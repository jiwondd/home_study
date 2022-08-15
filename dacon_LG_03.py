import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from csv import reader
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import time
import matplotlib as plt
from sklearn.decomposition import PCA

#.1 데이터
path='./_data/_dacon_LG/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'sample_submission.csv',index_col=0)
test_set=pd.read_csv(path+'test.csv',index_col=0) #예측할때 사용할거에요!!

# input_output 지정
x_train = train_set.filter(regex='X') 
y_train = train_set.filter(regex='Y')

scaler=MinMaxScaler()
scaler.fit(x_train)
x=scaler.transform(x_train)

x_train, x_test, y_train, y_test=train_test_split(x_train,y_train,shuffle=True,random_state=123,train_size=0.8)

kFold=KFold(n_splits=5, shuffle=True,random_state=123)

XG_parameters={'n_estimators':[100],
            'learning_rate':[0.001],
            'max_depth':[3],
            'gamma':[0],
            'min_child_weight':[1],
            'subsample':[0.1],
            'colsample_bytree':[0],
            'colsample_bylevel':[0],
            'colsample_bynode':[0],
            'reg_alpha':[2],
            'reg_lambda':[2]
           }

RF_parameters = [
    {'n_estimators':[100,200]},
    {'max_depth':[1,3,6,8,10]},
    {'min_samples_leaf':[1,2,3,5]},
    {'min_samples_split':[2,4,6,8]}
]

# 2. 모델구성
XG =XGBRegressor(random_state=123)
RF = RandomForestRegressor(random_state=123)
model=GridSearchCV(XG,XG_parameters,cv=kFold,n_jobs=8)

# 3. 훈련
import time
start=time.time()
model.fit(x_train,y_train)
end=time.time()

# 4. 평가, 예측
result=model.score(x_test,y_test)
print('model.score:',result)

# test_x = pd.read_csv('./test.csv').drop(columns=['ID'])
preds = model.predict(test_set)

for idx, col in enumerate(submission.columns):
    if col=='ID':
        continue
    submission[col] = preds[:,idx-1]
    
submission.to_csv(path + 'sample_submission.csv', index=True)