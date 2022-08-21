import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import math
import time
from icecream import ic
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold,\
    HalvingRandomSearchCV, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer


# 1. 데이터
filepath = './_data/dacon_travel/'
train = pd.read_csv(filepath+'train.csv', index_col=0)
test = pd.read_csv(filepath+'test.csv', index_col=0)

# print(train.head())
# print(train.info())
# print(train.isnull().sum())

# 결측치 TypeofContact 빼고 중간값으로 대체함, 데이터 수치들 보면 중간값이 제일 무난할거 같음--------------------
train['Age'].fillna(train['Age'].median(), inplace=True)
train['TypeofContact'].fillna('N', inplace=True) # N으로 채운 이유: 콘택 타입 없는 건 '없음'으로 주고 처리하기 위해
train['DurationOfPitch'].fillna(train['DurationOfPitch'].median(), inplace=True)
train['NumberOfFollowups'].fillna(train['NumberOfFollowups'].median(), inplace=True)
train['PreferredPropertyStar'].fillna(train['PreferredPropertyStar'].median(), inplace=True)
train['NumberOfTrips'].fillna(train['NumberOfTrips'].median(), inplace=True)
train['NumberOfChildrenVisiting'].fillna(train['NumberOfChildrenVisiting'].median(), inplace=True)
train['MonthlyIncome'].fillna(train['MonthlyIncome'].median(), inplace=True)

test['Age'].fillna(test['Age'].median(), inplace=True)
test['TypeofContact'].fillna('N', inplace=True)
test['DurationOfPitch'].fillna(test['DurationOfPitch'].median(), inplace=True)
test['NumberOfFollowups'].fillna(test['NumberOfFollowups'].median(), inplace=True)
test['PreferredPropertyStar'].fillna(test['PreferredPropertyStar'].median(), inplace=True)
test['NumberOfTrips'].fillna(test['NumberOfTrips'].median(), inplace=True)
test['NumberOfChildrenVisiting'].fillna(test['NumberOfChildrenVisiting'].median(), inplace=True)
test['MonthlyIncome'].fillna(test['MonthlyIncome'].median(), inplace=True)
# print(train.isnull().sum())
#-----------------------------------------------------------------------------------------------------------

# object타입 라벨인코딩--------------------
le = LabelEncoder()
idxarr = train.columns
idxarr = np.array(idxarr)

for i in idxarr:
      if train[i].dtype == 'object':
        train[i] = le.fit_transform(train[i])
        test[i] = le.fit_transform(test[i])
# print(train.info())
# ------------------------------------------

# 피처임포턴스 그래프 보기 위해 데이터프레임형태의 x_, y_ 놔둠 / 훈련용 넘파이어레이형태의 x, y 생성-----------
# x_ = train.drop(['ProdTaken','NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar'], axis=1) # 피처임포턴스로 확인한 중요도 낮은 탑3 제거
x_ = train.drop(['ProdTaken','NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar', 'MonthlyIncome', 'Designation'], axis=1)
# x_ = train.drop(['ProdTaken'], axis=1)
y_ = train['ProdTaken']
x = np.array(x_)
y = np.array(y_)
y = y.reshape(-1, 1) # y값 reshape 해야되서 x도 넘파이로 바꿔 훈련하는 것

# test = test.drop(['NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar'], axis=1) # 피처임포턴스로 확인한 중요도 낮은 탑3 제거
test = test.drop(['NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar', 'MonthlyIncome','Designation'], axis=1)
test = np.array(test)
# print(x.shape, y.shape)
#-----------------------------------------------------------------------------------------------------------

a3 =  [  89,  110,  120,  121,  137,  147,  172,  210,  211,  247,  310,
        365,  397,  398,  426,  448,  450,  567,  755,  802,  865,  935,
       1033, 1036, 1100, 1136, 1237, 1241, 1387, 1413, 1435, 1447, 1449,
       1461, 1471, 1521, 1540, 1652, 1665, 1722, 1782, 1919, 1936, 1947]

a6 = [   8,   10,   13,   33,   40,   53,   80,   85,  108,  127,  129,
        150,  151,  172,  178,  199,  206,  214,  233,  254,  256,  281,
        327,  335,  344,  355,  364,  367,  380,  420,  426,  438,  462,
        496,  498,  512,  575,  610,  631,  689,  711,  714,  718,  726,
        757,  761,  768,  777,  778,  790,  816,  827,  829,  834,  840,
        857,  871,  875,  904,  913,  946,  960,  971,  972,  994, 1007,
       1009, 1021, 1024, 1060, 1077, 1081, 1098, 1099, 1121, 1124, 1142,
       1182, 1208, 1237, 1245, 1259, 1289, 1296, 1314, 1322, 1337, 1344,
       1348, 1362, 1365, 1377, 1389, 1400, 1422, 1436, 1473, 1505, 1558,
       1591, 1598, 1606, 1610, 1621, 1631, 1641, 1677, 1715, 1724, 1726,
       1750, 1766, 1817, 1825, 1852, 1858, 1873, 1881, 1887, 1925, 1930]

a10 = [  90,  106,  129,  161,  177,  248,  302,  303,  316,  348,  425,
        552,  623,  698,  769,  873,  927,  987, 1055, 1076, 1175, 1325,
       1360, 1404, 1425, 1514, 1525, 1598, 1611, 1663, 1731, 1749, 1753,
       1768, 1828, 1844, 1853, 1933]

a13 = [  14,   59,   93,  105,  142,  167,  187,  203,  209,  218,  230,
        250,  265,  311,  314,  322,  342,  378,  447,  479,  503,  536,
        570,  592,  643,  662,  727,  729,  749,  768,  817,  819,  827,
        851,  852,  869,  912,  917,  944,  964,  995, 1020, 1097, 1101,
       1110, 1121, 1123, 1156, 1157, 1203, 1277, 1311, 1316, 1336, 1355,
       1363, 1380, 1398, 1417, 1474, 1486, 1502, 1507, 1518, 1526, 1536,
       1561, 1571, 1576, 1578, 1597, 1640, 1643, 1668, 1676, 1716, 1719,
       1739, 1750, 1783, 1791, 1818, 1822, 1823, 1856, 1870, 1887, 1899,
       1927]

for i in range(len(a3)): # DurationOfPitch
    x[a3[i]][3] = 20

# x[485][4] = np.nan # Occupation

# for i in range(len(a6)): # ProductPitched
#     x[a6[i]][6] = np.nan
    
for i in range(len(a10)): # NumberOfTrips
    x[a10[i]][10] = 5.7

# for i in range(len(a13)): # Designation
#     x[a13[i]][13] = np.nan
    
# outliers_printer(x)

parameters_xgb = {
            'n_estimators':[50,100,200,300,400,500],
            'learning_rate':[0.1,0.2,0.3,0.5,1,0.01,0.001],
            'max_depth':[None,2,3,4,5,6,7,8,9,10],
            'gamma':[0,1,2,3,4,5,7,10,100],
            'min_child_weight':[0,0.1,0.001,0.5,1,5,10,100],
            'subsample':[0,0.1,0.2,0.3,0.5,0.7,1],
            'reg_alpha':[0,0.1,0.01,0.001,1,2,10],
            'reg_lambda':[0,0.1,0.01,0.001,1,2,10],
              } 

parameters_rnf = [
    {'n_estimators':[100,200,300]},
    {'max_depth':[None,6,8,10,12]},
    {'min_samples_leaf':[3,5,7,10,11,13]},
    {'min_samples_split':[2,3,5,7,10]},
    {'n_jobs':[-1]}
]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=704, shuffle=True)
# print(np.unique(y_train, return_counts=True))

# 2. 모델
xgb = XGBRFClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
rnf = RandomForestClassifier(random_state=704) # 704 : 0.9053708439897699 / Designation 드랍하면 678 : 0.9053708439897699
# model = xgb
# model = rnf
# model = make_pipeline(MinMaxScaler(), HalvingRandomSearchCV(xgb, parameters_xgb, cv=5, n_jobs=-1, verbose=2))
model = make_pipeline(MinMaxScaler(), HalvingRandomSearchCV(rnf, parameters_rnf, cv=6, n_jobs=-1, verbose=2, random_state=704))
# model = make_pipeline(MinMaxScaler(), GridSearchCV(rnf, parameters_rnf, cv=5, n_jobs=-1, verbose=2))
# model = make_pipeline(MinMaxScaler(), xgb)
# model = make_pipeline(MinMaxScaler(), rnf)

start = time.time()
model.fit(x_train, y_train)
end = time.time()

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('스코어: ', results)
print('걸린 시간: ', end-start)

# 5. 제출 준비
y_submit = model.predict(test)

submission = pd.read_csv(filepath+'sample_submission.csv', index_col=0)
submission['ProdTaken'] = y_submit
submission.to_csv(filepath + 'sample_submission704.csv', index = True)

