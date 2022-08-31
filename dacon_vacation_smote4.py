# https://dacon.io/competitions/official/235959/overview/description

from cProfile import label
from tkinter import Y
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
from bayes_opt import BayesianOptimization


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
train_set = train_set.replace({'MaritalStatus' : 'Divorced'}, 'Single')
test_set = test_set.replace({'MaritalStatus' : 'Divorced'}, 'Single')
# train_set = train_set.replace({'MaritalStatus' : 'Unmarried'}, 'Married')
# test_set = test_set.replace({'MaritalStatus' : 'Unmarried'}, 'Married')

train_set['Age'].fillna(train_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
test_set['Age'].fillna(test_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
train_set['MonthlyIncome'].fillna(train_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
test_set['MonthlyIncome'].fillna(test_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
train_set['PreferredPropertyStar'].fillna(train_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
test_set['PreferredPropertyStar'].fillna(test_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)

train_set['Age']=np.round(train_set['Age'],0).astype(int)
test_set['Age']=np.round(test_set['Age'],0).astype(int)

train_set = train_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting','NumberOfFollowups'], axis=1)
test_set = test_set.drop(['NumberOfChildrenVisiting', 'NumberOfPersonVisiting','NumberOfFollowups'], axis=1)
train_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
test_set['TypeofContact'].fillna('Self Enquiry', inplace=True)

label=train_set['ProdTaken']
total_set=pd.concat((train_set,test_set)).reset_index(drop=True)
total_set=total_set.drop(['ProdTaken'],axis=1)
# print(total_set.shape) #(4888, 18)

total_set = pd.get_dummies(total_set)

imputer=IterativeImputer(random_state=123)
imputer.fit(total_set)
total_set=imputer.transform(total_set)

# scaler=QuantileTransformer()
scaler=MinMaxScaler()
scaler.fit(total_set)
x=scaler.transform(total_set)

train_set=total_set[:len(train_set)]
test_set=total_set[len(train_set):]

x=train_set
y=label

# smote=SMOTE(random_state=123)
# x,y=smote.fit_resample(x,y)
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
y_summit = model.predict(test_set)
submission['ProdTaken'] = y_summit
print('model.score:',result) 
# submission.to_csv('./_data/dacon_travel/sample_submission_93.csv', index=True)


# model.score: 0.8695652173913043 RF
# model.score: 0.8414322250639387 xgb
# model.score: 0.9258312020460358 RF smote
# model.score: 0.8849104859335039 
# model.score: 0.9332273449920508 <-트레인테스트전에 스모트
# model.score: 0.9411764705882353
# model.score: 0.9395866454689984 <- 'MonthlyIncome'
# model.score: 0.9411764705882353 <- 'NumberOfFollowups'
# model.score: 0.8746803069053708 <- 스케일러 전체셋에 하니까 점수가 맞음
# model.score: 0.8772378516624041 <- 'MonthlyIncome','NumberOfFollowups' 다시 살림
# model.score: 0.8849104859335039 <- 스모트 빼버림
# model.score: 0.8746803069053708 <-divorce,single/marrid,unmerride
# model.score: 0.8823529411764706 <-디보스&싱글/메리/언메리
# model.score: 0.8797953964194374 <-싱글/메리&언메리/디볼스
# model.score: 0.8823529411764706 <-민맥스
# model.score: 0.8721227621483376 <-민맥스+스모트
# model.score: 0.9300476947535771 <-스모트 위치변경 / 서브미션 0.880
# model.score: 0.931637519872814 <-전처리변경
