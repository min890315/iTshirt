# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 23:43:14 2020

@author: min89
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib

## 데이터 불러오기 (TARGET 열이 분류의 기준이 되며, 1은 불만을 가진 고객이고 0은 만족한 고객)
cust_df = pd.read_csv("santander-customer-satisfaction/train.csv",encoding='latin-1') # encoding은 설정하지 않아도 읽을 수 있음
print('dataset shape:', cust_df.shape)
print(cust_df.head(3))
print(cust_df.info())
print(cust_df.describe())

print(cust_df['TARGET'].value_counts())

unsatisfied_cust = cust_df[cust_df['TARGET']==1].TARGET.count()
total_cust = cust_df.TARGET.count()

print("불만족 고객의 비율: {0}".format(unsatisfied_cust/total_cust))


### 데이터 전처리
## describe로 확인해보면 var3열에 -999999라는 값이 있음(분석에 너무 심한 오차를 줄 수 있으므로 다른 값으로 변환)
cust_df['var3'].replace(-999999,2, inplace=True) # var3열의 -999999를 가장 많은 비율을 차지하는 2로 바꿈 (inplace=True를 사용하여 다른 변수에 지정하지 않고 원본을 바꿈)
cust_df.drop('ID',axis=1 , inplace=True) # ID열은 고객 식별자 (분석에 필요없으므로 drop으로 제거)
print(cust_df.describe())

## 피처 세트와 레이블 세트분리. 레이블 컬럼은 DataFrame의 맨 마지막에 위치해 컬럼 위치 -1로 분리
X_features = cust_df.drop("TARGET", axis=1)
y_labels = cust_df["TARGET"]
print('피처 데이터 shape:{0}'.format(X_features.shape))

### 학습과 테스트 데이터 세트를 8:2 비율로 분리
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels,
                                                    test_size=0.2, random_state=0)
train_cnt = y_train.count()
test_cnt = y_test.count()
print('학습 세트 Shape:{0}, 테스트 세트 Shape:{1}'.format(X_train.shape , X_test.shape))

## 원본의 비율과 맞게 분리되었는지 레이블의 비율을 비교
print(' 학습 세트 레이블 값 분포 비율')
print(y_train.value_counts()/train_cnt)
print('\n 테스트 세트 레이블 값 분포 비율')
print(y_test.value_counts()/test_cnt)
## 분리되기 전과 거의 비슷한 비율로 분할됨(테스트 세트의 1이 0.39가 아닌 0.41의 비율을 차지하지만 오차범위 이내)


### XGBoost를 이용한 모델 학습과 하이퍼 파라미터 튜닝
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# n_estimators는 500으로, random state는 예제 수행 시마다 동일 예측 결과를 위해 설정. 
xgb_clf = XGBClassifier(n_estimators=500, random_state=156)

# 성능 평가 지표를 auc로, 조기 중단 파라미터는 100으로 설정하고 학습 수행
#조기 중단을 위한 성능지표는 auc(Area under the curve, ROC curve의 아래쪽 영역 넓이)
xgb_clf.fit(X_train, y_train, early_stopping_rounds=100,
            eval_metric="auc", eval_set=[(X_train, y_train), (X_test, y_test)])

xgb_roc_score = roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:,1],average='macro')
print('ROC AUC: {0:.4f}'.format(xgb_roc_score))
# 최고의 성능을 나타내는 것은 178번째 학습


## GridSearchCV를 사용한 하이퍼 파라미터 조정
from sklearn.model_selection import GridSearchCV

# 하이퍼 파라미터 테스트의 수행 속도를 향상시키기 위해 n_estimators를 100으로 감소
xgb_clf = XGBClassifier(n_estimators=100)

# 적용할 하이퍼 파라미터
params = {'max_depth':[5, 7] , 'min_child_weight':[1,3] ,'colsample_bytree':[0.5, 0.75] }
# min_child_samples: 과적합을 개선하기위한 주요 파라미터, 큰 값으로 설정하면 트리가 깊어지는 것을 방지
# max_depth: 깊이의 크기를 제한, 위의 두 파라미터와 결합하여 과적합 개선
# colsample_bytree: 개별 의사결정나무 모형에 사용될 변수갯수를 지정. 보통 0.5 ~ 1를 사용하며 기본값은 1


# cv(교차검증)는 3으로 지정 
gridcv = GridSearchCV(xgb_clf, param_grid=params, cv=3)
gridcv.fit(X_train, y_train, early_stopping_rounds=30, eval_metric="auc",
           eval_set=[(X_train, y_train), (X_test, y_test)])

print('GridSearchCV 최적 파라미터:',gridcv.best_params_) 
# GridSearchCV 최적 파라미터: {'colsample_bytree': 0.5, 'max_depth': 5, 'min_child_weight': 1}

xgb_roc_score = roc_auc_score(y_test, gridcv.predict_proba(X_test)[:,1], average='macro')
print('ROC AUC: {0:.4f}'.format(xgb_roc_score))
# ROC AUC: 0.8461


### 찾아낸 최적의 조건을 사용하여 한번 더 학습 진행
# n_estimators는 1000으로 증가시키고, learning_rate=0.02로 감소, reg_alpha=0.03으로 추가함. (하이퍼 파라미터를 조금 더 추가)
xgb_clf = XGBClassifier(n_estimators=1000, random_state=156, learning_rate=0.02, max_depth=7,\
                        min_child_weight=1, colsample_bytree=0.75, reg_alpha=0.03)
    

# evaluation metric을 auc로, early stopping은 200 으로 설정하고 학습 수행. 
xgb_clf.fit(X_train, y_train, early_stopping_rounds=200, 
            eval_metric="auc",eval_set=[(X_train, y_train), (X_test, y_test)])

xgb_roc_score = roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:,1],average='macro')
print('ROC AUC: {0:.4f}'.format(xgb_roc_score))


## 피처의 중요도 시각화
from xgboost import plot_importance
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1,figsize=(10,8))
plot_importance(xgb_clf, ax=ax , max_num_features=20,height=0.4)






