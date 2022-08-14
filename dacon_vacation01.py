# https://dacon.io/competitions/official/235959/overview/description

from cProfile import label
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
train_set = train_set.drop(['NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar'], axis=1)
test_set = test_set.drop(['NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar'], axis=1)

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

imputer=IterativeImputer()
imputer.fit(total_set)
total_set=imputer.transform(total_set)

train_set=total_set[:len(train_set)]
test_set=total_set[len(train_set):]

x=train_set
y=label

# x=pd.DataFrame(x)
# print(x.isnull().sum()) 
# 임퓨터 제대로 들어갔는지 보려고 넘파이를 데이터프레임으로 잠깐 바꿔봄

scaler=RobustScaler()
scaler.fit(x)
x=scaler.transform(x)

x_train, x_test, y_train, y_test=train_test_split(x,y,shuffle=True,random_state=42,train_size=0.9,stratify=y)

# scaler=StandardScaler()
# scaler.fit(x)
# x=scaler.transform(x)

# lda=LinearDiscriminantAnalysis() 
# lda.fit(x,y)
# x=lda.transform(x)


# x_train, x_test, y_train, y_test=train_test_split(x,y,shuffle=True,random_state=134,train_size=0.8,stratify=y)

kFold=StratifiedKFold(n_splits=5, shuffle=True,random_state=42)

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
XG =XGBClassifier(random_state=123)
RF = RandomForestClassifier(random_state=42)
model=GridSearchCV(RF,RF_parameters,cv=kFold,n_jobs=8)

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
result=model.score(x_test,y_test)
print('model.score:',result) 
best_params=model.best_params_
print('best_params : ', best_params )

#5. 데이터 summit
y_summit = model.predict(test_set)
y_summit = y_summit.flatten()                 
y_summit = np.where(y_summit > 0.55, 1 , 0)   

submission['ProdTaken'] = y_summit
print(submission)
submission.to_csv('./_data/dacon_travel/sample_submission.csv', index=True)


# model.score: 0.8695652173913043 RF
# model.score: 0.8414322250639387 xgb


'''
model.score: 0.80306905370844 
model.score: 0.8061224489795918 <-이상치 3개빼고 스탠다드
model.score: 0.8061224489795918 <-민맥스로 바꿔봄
model.score: 0.8775510204081632 <-RF로 바꿔봄 (랜덤 123)
model.score: 0.8877551020408163 <-랜덤 42

'''
