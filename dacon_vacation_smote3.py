# https://dacon.io/competitions/official/235959/overview/description

from cProfile import label
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from csv import reader
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


#.1 데이터
path='./_data/dacon_travel/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'sample_submission.csv',index_col=0)
test_set=pd.read_csv(path+'test.csv',index_col=0) #예측할때 사용할거에요!!
# print(train_set.shape) (1459, 10)
# print(test_set.shape) (715, 9)

train_set = train_set.replace({'Gender' : 'Fe Male'}, 'Female')
test_set = test_set.replace({'Gender' : 'Fe Male'}, 'Female')
train_set = train_set.replace({'Occupation':'Free Lancer'}, 'Small Business')
test_set = test_set.replace({'Occupation':'Free Lancer'}, 'Small Business')

train_set = train_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting', 'MonthlyIncome', 'NumberOfFollowups'], axis=1)
test_set = test_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting', 'MonthlyIncome', 'NumberOfFollowups'], axis=1)
train_set['TypeofContact'].fillna('N', inplace=True)

label=train_set['ProdTaken']
total_set=pd.concat((train_set,test_set)).reset_index(drop=True)
total_set=total_set.drop(['ProdTaken'],axis=1)
# print(total_set.shape) #(4888, 18)

total_set = pd.get_dummies(total_set)

imputer=IterativeImputer(random_state=123)
imputer.fit(total_set)
total_set=imputer.transform(total_set)

train_set=total_set[:len(train_set)]
test_set=total_set[len(train_set):]

x=train_set
y=label
# print(np.unique(y, return_counts=True))
# (array([0, 1], dtype=int64), array([1572,  383], dtype=int64))

scaler=QuantileTransformer()
scaler.fit(x)
x=scaler.transform(x)

smote=SMOTE(random_state=123)
x,y=smote.fit_resample(x,y)
# print(np.unique(y, return_counts=True))

x_train, x_test, y_train, y_test=train_test_split(x,y,shuffle=True,random_state=123,train_size=0.8,stratify=y)

kFold=StratifiedKFold(shuffle=True,random_state=123)

# 2. 모델구성
# model=RandomForestClassifier(random_state=123)
model=CatBoostClassifier(random_state=123)

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
result=model.score(x_test,y_test)
# print('model.score:',result) 

#5. 데이터 summit
model.fit(x,y)
result2=model.score(x,y)
y_summit = model.predict(test_set)
submission['ProdTaken'] = y_summit
print('model.score:',result) 
print('results2:',result2)
# submission.to_csv('./_data/dacon_travel/sample_submission3.csv', index=True)


# model.score: 0.8695652173913043 RF
# model.score: 0.8414322250639387 xgb
# model.score: 0.9258312020460358 RF smote
# model.score: 0.8849104859335039 
# model.score: 0.9332273449920508 <-트레인테스트전에 스모트
# model.score: 0.9411764705882353