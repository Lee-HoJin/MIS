import pandas as pd
from scipy.stats import f_oneway
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 데이터 불러오기 Data Loading
data = pd.read_csv("./dataset.csv", sep=",", low_memory=False)

# 사용하는 모든 변수들
variables = ['MisleadingHealthInfo',
             'Age',
             'BirthGender',
             'MaritalStatus',
             'IncomeFeelings',
             'SmokeNow',
             'TimesModerateExercise',
             'SocMed_MakeDecisions',
             'SocMed_DiscussHCP',
             'SocMed_SharedPers',
             'SocMed_SharedGen',
             'SocMed_Interacted',
             'SocMed_WatchedVid']


###################################################
####### 수치화 Converting to Numeric Values #######
###################################################

data['MisleadingHealthInfo'] = data['MisleadingHealthInfo'].replace({
    'I do not use social media': -1,
    'A little': 0,
    'Some': 1,
    'A lot': 2
})

data['Age'] = pd.to_numeric(data['Age'], errors='coerce')

## 여성이 1, 남성이 0 (여성 응답자가 더 많음)
data['BirthGender'] = np.where(data['BirthGender'] == 'Female', 1, 0)

data['MaritalStatus'] = data['MaritalStatus'].replace({
    'Single, never been married' : 0,
    'Separated' : 0,
    'Widowed': 0,
    'Divorced' : 0,
    'Living as married or living with a romantic partner': 0,
    'Married': 1
})

data['IncomeFeelings'] = data['IncomeFeelings'].replace({
    'Living comfortably on present income' : 3,
    'Getting by on present income' : 2,
    'Finding it difficult on present income': 1,
    'Finding it very difficult on present income' : 0,
})


# N5
data['SmokeNow'] = data['SmokeNow'].replace({
    'Not at all' : 0,
    'Some days' : 1,
    'Every day' : 2
})

# M1
data['TimesModerateExercise'] = data['TimesModerateExercise'].replace({
        'None' : 0,
        '1 day per week' : 1,
        '2 days per week' : 2,
        '3 days per week' : 3,
        '4 days per week' : 4,
        '5 days per week' : 5,
        '6 days per week' : 6,
        '7 days per week' : 7,
    })

B14 = ['SocMed_MakeDecisions', 'SocMed_DiscussHCP', 'SocMed_TrueFalse', 'SocMed_SameViews']
for target in B14 :
    data[target] = data[target].replace({
        'Strongly disagree' : 0,
        'Somewhat disagree' : 1,
        'Somewhat agree' : 2,
        'Strongly agree' : 3,
        'Strongle agree' : 3,
        'Strongle agree' : 3
    })

B12 = ['SocMed_Visited', 'SocMed_SharedPers', 'SocMed_SharedGen', 'SocMed_Interacted', 'SocMed_WatchedVid']
for target in B12 :
    data[target] = data[target].replace({
        'Never' : 0,
        'Less than once a month' : 1,
        'A few times a month' : 2,
        'At least once a week' : 3,
        'Almost every day' : 4
    })

###################################################
########### 데이터 전처리 Preprocessing ###########
###################################################

y = 'SocMed_MakeDecisions' 

for target in ['Age', 'IncomeFeelings']:
    # target 컬럼에서 숫자가 아닌 값들로 이루어진 행을 제거
    data = data[~data[target].apply(lambda x: isinstance(x, str))]

    # 결측치 제거
    data = data.dropna(subset=[target])
    
    try:
        # 'typecasting' 변수에 대해 int 변환 시도
        data[target] = data[target].astype(int)
    except ValueError as e:
        print(f"Error casting {target}: {e}")
        # 변환 실패한 변수는 건너뜀

# 데이터 스케일링
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Age', 'IncomeFeelings']])

# PCA 적용
pca = PCA(n_components=1)  # 하나의 주성분만 생성
data['Age_Income_PC1'] = pca.fit_transform(scaled_data)
# print(pca.components_)

X_model_1 = ['Age_Income_PC1',
             'MaritalStatus',
             'BirthGender', 
             'SocMed_DiscussHCP',
             'SmokeNow',
             'TimesModerateExercise',
            ]
X_model_2 = X_model_1 + ['SocMed_SharedPers', 'SocMed_SharedGen', 'SocMed_Interacted', 'SocMed_WatchedVid']
X_model_3 = X_model_2 + ['MisleadingHealthInfo']

models = [X_model_1, X_model_2, X_model_3]

# 'Social Media' 사용자들의 응답만 남기기 위함
data['MisleadingHealthInfo'] = data['MisleadingHealthInfo'].replace(-1, np.nan)

for target in ['MisleadingHealthInfo'] + X_model_3 + [y]:
    # target 컬럼에서 숫자가 아닌 값들로 이루어진 행을 제거
    data = data[~data[target].apply(lambda x: isinstance(x, str))]

    # 결측치 제거
    data = data.dropna(subset=[target])
    
    try:
        # 'typecasting' 변수에 대해 int 변환 시도
        data[target] = data[target].astype(int)
    except ValueError as e:
        print(f"Error casting {target}: {e}")
        # 변환 실패한 변수는 건너뜀


###################################################
########## 회귀 분석 파트 Regression Part ##########
###################################################

# 성능 결과를 저장할 딕셔너리
model_performance = {
    'Model': [],
    'R-squared': [],
    'MSE': [],
    'CV std': [],
    'Pseudo R-squared': [],
    'TensorFlow': []
}

# 각 모델에 대해 반복문 실행 후 성능 기록
for i, X in enumerate(models, 1):
    # y 변수는 1D 배열로 변환
    X_train, X_test, y_train, y_test = train_test_split(data[X], data[y], test_size=0.2, random_state=42)

    # 선형 회귀 모델 생성 및 학습
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # 예측
    y_pred = linear_model.predict(X_test)

    # 평가
    mse = mean_squared_error(y_test, y_pred)
    r_squared = linear_model.score(X_test, y_test)

    # 교차 검증 점수 계산 (평가 지표는 Mean Squared Error)
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    cv_scores = cross_val_score(linear_model, X_train, y_train, cv=5, scoring=mse_scorer)

    # 순서형 로지스틱 회귀 Ordered Logistic Regression
    order_logit_model = OrderedModel(
        data[y].astype(int),
        data[X],
        distr='logit'
    )
    order_logit_result = order_logit_model.fit(method='bfgs')
    # print(order_logit_result.summary())

    # 로지스틱 회귀 모델의 로그 우도 값
    log_likelihood_model = order_logit_result.llf  # 모델의 로그 우도 값

    # 비교 모델 (보통 상수만 있는 모델)의 로그 우도 값
    log_likelihood_null = order_logit_result.llnull  # 상수만 포함된 모델의 로그 우도 값

    # McFadden's R-squared 계산
    pseudo_r_squared = 1 - (log_likelihood_model / log_likelihood_null)

    print(f"McFadden's Pseudo R-squared: {pseudo_r_squared:.4f}")

    #### 텐서 플로우 활용
    tensor_y = data[y]

    # 데이터 스케일링
    scaler = StandardScaler()  
    tensor_x = scaler.fit_transform(data[X].values)    
    
    X_train, X_test, y_train, y_test = train_test_split(tensor_x, tensor_y, test_size=0.2, random_state=42)
        
    # 모델 설정 (다중 클래스 분류 모델)
    model = Sequential()

    # 입력층과 은닉층 추가
    model.add(Dense(32, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)))  # 은닉층
    model.add(Dropout(0.2))
    
    # 출력층 - 4개의 클래스 (Strongly disagree, Somewhat disagree, Somewhat agree, Strongly agree)
    model.add(Dense(4, activation='softmax'))  # Softmax로 4개 클래스 분류

    # 컴파일 (다중 클래스 분류)
    optimizer = Adam(learning_rate = 0.05)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # y_train과 y_test를 원-핫 인코딩
    y_train_encoded = to_categorical(y_train, num_classes=4)
    y_test_encoded = to_categorical(y_test, num_classes=4)
    
    # 조기 종료 설정 (성능이 개선되지 않으면 학습 중단)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 모델 학습
    history = model.fit(X_train, y_train_encoded,
              epochs = 20,
              batch_size = 32,
              verbose = 0,
              validation_data=(X_test, y_test_encoded),
              callbacks=[early_stopping])
    
    # 평가
    loss, accuracy = model.evaluate(X_test, y_test_encoded)
    print(f'Test Accuracy: {accuracy}')
    
    # 성능 기록
    model_performance['Model'].append(f'Model {i}')
    model_performance['R-squared'].append(r_squared)
    model_performance['MSE'].append(mse)
    model_performance['CV std'].append(np.std(cv_scores))
    model_performance['Pseudo R-squared'].append(pseudo_r_squared)
    model_performance['TensorFlow'].append(accuracy)
    
# 성능 결과 출력
performance_df = pd.DataFrame(model_performance)
print(performance_df)