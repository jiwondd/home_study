import pandas as pd
import random
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV

# 랜덤고정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42)

# 1. 데이터
path='./_data/_dacon_LG/'
train_df = pd.read_csv(path+'train.csv')

train_x = train_df.filter(regex='X') # Input : X Featrue
train_y = train_df.filter(regex='Y') # Output : Y Feature

pca=PCA(n_components=14)
train_x=pca.fit_transform(train_x)
train_y=pca.fit_transform(train_y)

scaler=StandardScaler()
scaler.fit(train_x)
x=scaler.transform(train_x)

x_train, x_test, y_train, y_test=train_test_split(train_y,train_y,shuffle=True,random_state=123,train_size=0.8)

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
model=GridSearchCV(RF,RF_parameters,cv=kFold,n_jobs=8)

# 3. 훈련
import time
start=time.time()
model.fit(x_train,y_train)
end=time.time()

# 4. 평가
result=RF.score(train_x,train_y)
print('model.score:',result)


# 5. submission
test_x = pd.read_csv(path+'test.csv').drop(columns=['ID'])

preds = RF.predict(test_x)
print('Done.')

submit = pd.read_csv(path+'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = preds[:,idx-1]
print('Done.')

submit.to_csv('./submit.csv', index=False)


#This RandomForestRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

