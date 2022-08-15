import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from csv import reader
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,r2_score
from sklearn.feature_selection import SelectFromModel
import time
import matplotlib as plt

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
x = train_set.drop(['Y_01', 'Y_02', 'Y_03', 'Y_04', 'Y_05', 'Y_06', 'Y_07',
       'Y_08', 'Y_09', 'Y_10', 'Y_11', 'Y_12', 'Y_13', 'Y_14'], axis=1)
y = train_set.drop(['X_01', 'X_02', 'X_03', 'X_04', 'X_05', 'X_06', 'X_07', 'X_08', 'X_09',
       'X_10', 'X_11', 'X_12', 'X_13', 'X_14', 'X_15', 'X_16', 'X_17', 'X_18',
       'X_19', 'X_20', 'X_21', 'X_22', 'X_23', 'X_24', 'X_25', 'X_26', 'X_27',
       'X_28', 'X_29', 'X_30', 'X_31', 'X_32', 'X_33', 'X_34', 'X_35', 'X_36',
       'X_37', 'X_38', 'X_39', 'X_40', 'X_41', 'X_42', 'X_43', 'X_44', 'X_45',
       'X_46', 'X_47', 'X_48', 'X_49', 'X_50', 'X_51', 'X_52', 'X_53', 'X_54',
       'X_55', 'X_56'],axis=1)
# print(x.shape,y.shape) (39607, 56) (39607, 14)

scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)

x_train, x_test, y_train, y_test=train_test_split(x,y,shuffle=True,random_state=123,train_size=0.8)

kFold=KFold(n_splits=5, shuffle=True,random_state=123)


# 2. 모델구성
model=XGBRegressor(random_state=100,
                    n_estimator=100,
                    learnig_rate=0.3,
                    max_depth=6,
                    gamma=0)

# 3. 훈련
import time
start=time.time()
model.fit(x_train,y_train,early_stopping_rounds=100,
          eval_set=[(x_test,y_test)],
          eval_metric='rmse'
          )
end=time.time()

# 4. 평가
result=model.score(x_test,y_test)
print('model.score:',result) 
y_predict=model.predict(x_test)
r2=r2_score(y_test, y_predict)
print('진짜 최종 test 점수 : ' , r2)
print('걸린시간:',np.round(end-start,2))
print('---------------------------------')
print(model.feature_importances_)
thresholds=model.feature_importances_
print('---------------------------------')

y_submit = model.predict(test_set)
submission = pd.read_csv(path+'sample_submission.csv', index_col=0)

for idx, col in enumerate(submission.columns):
    if col=='ID':
        continue
    submission[col] = y_submit[:,idx-1]
    
submission.to_csv(path + 'sample_submission.csv', index=True)

for thresh in thresholds :
    selection=SelectFromModel(model, threshold=thresh, prefit=True)
    
    select_x_train=selection.transform(x_train)
    select_x_test=selection.transform(x_test)
    print(select_x_train.shape,select_x_test.shape)
    
    selection_model=XGBRegressor(n_jobs=-1,
                                 random_state=100,
                                 n_estimators=100,
                                 learning_rate=0.3,
                                 max_depth=6,
                                 gamma=0)
    selection_model.fit(select_x_train,y_train)
    y_predict=selection_model.predict(select_x_test)
    score=r2_score(y_test,y_predict)
    print("Thresh=%.3f,n=%d, r2:%.2f%%"
          #소수점3개까지,정수,소수점2개까지
          %(thresh,select_x_train.shape[1],score*100))

'''
model.score: 0.05019207934706029
진짜 최종 test 점수 :  0.05019207934706029
걸린시간: 43.9
---------------------------------
[0.00688153 0.00594767 0.01747787 0.         0.01854588 0.01620303 
 0.03601923 0.01449645 0.0393909  0.01454247 0.02249546 0.01638962 
 0.02289915 0.01839718 0.01897218 0.01435428 0.02051959 0.01686904 
 0.02477584 0.02044764 0.02305446 0.01889843 0.         0.01635848 
 0.01635997 0.01756629 0.01482338 0.01735932 0.01467927 0.02021507 
 0.01429307 0.03870756 0.01565562 0.01535921 0.01356271 0.01460679 
 0.01475617 0.015744   0.01871374 0.01696057 0.02227696 0.01915209 
 0.01700035 0.0179299  0.01731339 0.04050975 0.         0.
 0.03339878 0.01669573 0.01700228 0.01732584 0.01965705 0.01757091 
 0.01592686 0.02494099]
---------------------------------

0,1,49 빼볼까
(31685, 51) (7922, 51)
Thresh=0.007,n=51, r2:1.86%
(31685, 52) (7922, 52)
Thresh=0.006,n=52, r2:2.11%
(31685, 25) (7922, 25)
Thresh=0.017,n=25, r2:2.11%
(31685, 56) (7922, 56)
Thresh=0.000,n=56, r2:2.11%
(31685, 20) (7922, 20)
Thresh=0.019,n=20, r2:1.24%
(31685, 37) (7922, 37)
Thresh=0.016,n=37, r2:1.58%
(31685, 4) (7922, 4)
Thresh=0.036,n=4, r2:-2.20%
(31685, 47) (7922, 47)
Thresh=0.014,n=47, r2:1.34%
(31685, 2) (7922, 2)
Thresh=0.039,n=2, r2:-4.55%
(31685, 46) (7922, 46)
Thresh=0.015,n=46, r2:0.95%
(31685, 10) (7922, 10)
Thresh=0.022,n=10, r2:0.70%
(31685, 34) (7922, 34)
Thresh=0.016,n=34, r2:1.70%
(31685, 9) (7922, 9)
Thresh=0.023,n=9, r2:0.75%
(31685, 21) (7922, 21)
Thresh=0.018,n=21, r2:1.26%
(31685, 17) (7922, 17)
Thresh=0.019,n=17, r2:1.18%
(31685, 48) (7922, 48)
Thresh=0.014,n=48, r2:1.73%
(31685, 12) (7922, 12)
Thresh=0.021,n=12, r2:1.30%
(31685, 32) (7922, 32)
Thresh=0.017,n=32, r2:1.55%
(31685, 7) (7922, 7)
Thresh=0.025,n=7, r2:-0.08%
(31685, 13) (7922, 13)
Thresh=0.020,n=13, r2:1.13%
(31685, 8) (7922, 8)
Thresh=0.023,n=8, r2:-0.13%
(31685, 18) (7922, 18)
Thresh=0.019,n=18, r2:1.02%
(31685, 56) (7922, 56)
Thresh=0.000,n=56, r2:2.11%
(31685, 36) (7922, 36)
Thresh=0.016,n=36, r2:1.55%
(31685, 35) (7922, 35)
Thresh=0.016,n=35, r2:1.67%
(31685, 24) (7922, 24)
Thresh=0.018,n=24, r2:1.40%
(31685, 42) (7922, 42)
Thresh=0.015,n=42, r2:1.55%
(31685, 26) (7922, 26)
Thresh=0.017,n=26, r2:2.16%
(31685, 44) (7922, 44)
Thresh=0.015,n=44, r2:1.66%
(31685, 14) (7922, 14)
Thresh=0.020,n=14, r2:0.97%
(31685, 49) (7922, 49)
Thresh=0.014,n=49, r2:1.84%
(31685, 3) (7922, 3)
Thresh=0.039,n=3, r2:-3.52%
(31685, 40) (7922, 40)
Thresh=0.016,n=40, r2:1.39%
(31685, 41) (7922, 41)
Thresh=0.015,n=41, r2:1.24%
(31685, 50) (7922, 50)
Thresh=0.014,n=50, r2:1.51%
(31685, 45) (7922, 45)
Thresh=0.015,n=45, r2:1.20%
(31685, 43) (7922, 43)
Thresh=0.015,n=43, r2:0.98%
(31685, 39) (7922, 39)
Thresh=0.016,n=39, r2:1.70%
(31685, 19) (7922, 19)
Thresh=0.019,n=19, r2:0.86%
(31685, 31) (7922, 31)
Thresh=0.017,n=31, r2:1.72%
(31685, 11) (7922, 11)
Thresh=0.022,n=11, r2:0.83%
(31685, 16) (7922, 16)
Thresh=0.019,n=16, r2:1.20%
(31685, 30) (7922, 30)
Thresh=0.017,n=30, r2:1.81%
(31685, 22) (7922, 22)
Thresh=0.018,n=22, r2:1.21%
(31685, 28) (7922, 28)
Thresh=0.017,n=28, r2:1.83%
(31685, 1) (7922, 1)
Thresh=0.041,n=1, r2:-0.06%
(31685, 56) (7922, 56)
Thresh=0.000,n=56, r2:2.11%
(31685, 56) (7922, 56)
Thresh=0.000,n=56, r2:2.11%
(31685, 5) (7922, 5)
Thresh=0.033,n=5, r2:-1.15%
(31685, 33) (7922, 33)
Thresh=0.017,n=33, r2:1.63%
(31685, 29) (7922, 29)
Thresh=0.017,n=29, r2:2.01%
(31685, 27) (7922, 27)
Thresh=0.017,n=27, r2:1.91%
(31685, 15) (7922, 15)
Thresh=0.020,n=15, r2:1.13%
(31685, 23) (7922, 23)
Thresh=0.018,n=23, r2:1.43%
(31685, 38) (7922, 38)
Thresh=0.016,n=38, r2:1.75%
(31685, 6) (7922, 6)
Thresh=0.025,n=6, r2:-0.63%
'''