# https://dacon.io/competitions/official/235959/overview/description

from cProfile import label
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from csv import reader
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

#.1 데이터
path='./_data/dacon_travel/'
train_set=pd.read_csv(path+'train.csv',index_col=0)
submission=pd.read_csv(path+'sample_submission.csv',index_col=0)
test_set=pd.read_csv(path+'test.csv',index_col=0) #예측할때 사용할거에요!!

# print(train_set.shape) (1459, 10)
# print(test_set.shape) (715, 9)
train_set = train_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting', 'OwnCar', 'MonthlyIncome', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)
test_set = test_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting', 'OwnCar', 'MonthlyIncome', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)
train_set['TypeofContact'].fillna('N', inplace=True)

label=train_set['ProdTaken']
total_set=pd.concat((train_set,test_set)).reset_index(drop=True)
total_set=total_set.drop(['ProdTaken'],axis=1)
# print(total_set.shape) #(4888, 18)

le = LabelEncoder()
cols = ('TypeofContact','Occupation','Gender','ProductPitched','MaritalStatus','Designation')
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(total_set[c].values)) 
    total_set[c] = lbl.transform(list(total_set[c].values))

imputer=IterativeImputer(random_state=42)
imputer.fit(total_set)
total_set=imputer.transform(total_set)

train_set=total_set[:len(train_set)]
test_set=total_set[len(train_set):]

x=train_set
y=label

scaler=QuantileTransformer(output_distribution='normal')
x=scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x,y,shuffle=True,random_state=42,train_size=0.8,stratify=y)

kFold=StratifiedKFold(n_splits=5, shuffle=True,random_state=42)

# 2. 모델구성
model=RandomForestClassifier(random_state=42)

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
result=model.score(x_test,y_test)
print('model.score:',result) 

#5. 데이터 summit
y_summit = model.predict(test_set)
submission['ProdTaken'] = y_summit
submission.to_csv('./_data/dacon_travel/sample_submission2.csv', index=True)


'''

model.score: 0.80306905370844 
model.score: 0.8061224489795918 <-이상치 3개빼고 스탠다드
model.score: 0.8061224489795918 <-민맥스로 바꿔봄
model.score: 0.8775510204081632 <-RF로 바꿔봄 (랜덤 123)
model.score: 0.8877551020408163 <-랜덤 42
model.score: 0.8979591836734694 <-컨택타입을 n으로 바꿔봄
model.score: 0.8775510204081632 <-interpolate 
model.score: 0.8928571428571429 <-중위값
model.score: 0.8979591836734694 <-QuantileTransformer 랜포디폴트
model.score: 0.9183673469387755 <-xgb로 변경
model.score: 0.8877551020408163 <-컬럼더빼봄(xg)
model.score: 0.9132653061224489 <-컬럼더빼봄(RF)
model.score: 0.9002557544757033 <-랜덤스테이트 123...
'''
