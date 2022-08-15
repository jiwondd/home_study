import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from csv import reader
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
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

# print(train_set.shape,test_set.shape) #(39607, 70) (39608, 56)
# print(train_set.info())
# print(train_set.isnull().sum())
# print(test_set.isnull().sum())  
# 결측치가 없다고라?

# x=train_set[:,55]
# y=train_set[56:]
# print(x.shape,y.shape) 슬라이싱 안되네ㅎ
x = train_set.drop(['X_10','X_11','Y_01', 'Y_02', 'Y_03', 'Y_04', 'Y_05', 'Y_06', 'Y_07',
       'Y_08', 'Y_09', 'Y_10', 'Y_11', 'Y_12', 'Y_13', 'Y_14'], axis=1)
y = train_set.drop(['X_01', 'X_02', 'X_03', 'X_04', 'X_05', 'X_06', 'X_07', 'X_08', 'X_09',
       'X_10', 'X_11', 'X_12', 'X_13', 'X_14', 'X_15', 'X_16', 'X_17', 'X_18',
       'X_19', 'X_20', 'X_21', 'X_22', 'X_23', 'X_24', 'X_25', 'X_26', 'X_27',
       'X_28', 'X_29', 'X_30', 'X_31', 'X_32', 'X_33', 'X_34', 'X_35', 'X_36',
       'X_37', 'X_38', 'X_39', 'X_40', 'X_41', 'X_42', 'X_43', 'X_44', 'X_45',
       'X_46', 'X_47', 'X_48', 'X_49', 'X_50', 'X_51', 'X_52', 'X_53', 'X_54',
       'X_55', 'X_56'],axis=1)

test_set=test_set.drop(['X_10','X_11'], axis=1)
# print(x.shape) (39607, 54)

pca=PCA(n_components=54)
x=pca.fit_transform(x)

scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)

x_train, x_test, y_train, y_test=train_test_split(x,y,shuffle=True,random_state=123,train_size=0.8)

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

# 4. 평가, 예측
result=model.score(x_test,y_test)
print('model.score:',result) 

# 5. submmit 만들기 /rmse 계산하기

y_submit = model.predict(test_set)
submission = pd.read_csv(path+'sample_submission.csv', index_col=0)

for idx, col in enumerate(submission.columns):
    if col=='ID':
        continue
    submission[col] = y_submit[:,idx-1]
    
submission.to_csv(path + 'sample_submission.csv', index=True)

# model.score: 0.021052058672732903
# model.score: 0.015413729345914808 x10,x11번 칼럼 빼고
# model.score: 0.06695841874734547 랜포
# model.score: 0.06695841874734547
# model.score: -590.7168346288969 <-XG
# model.score: -598.2595822648194<-랜덤스테이트 123-1234
# model.score: 0.04734622864008282pca 추가해봄+랜포 
