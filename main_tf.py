import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

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
             'IncomeFeelings',
             'BirthGender',
             'MaritalStatus',
             'SocMed_MakeDecisions',
             'SocMed_DiscussHCP',
             'SocMed_TrueFalse',
             'SocMed_SameViews',
             'SocMed_SharedPers',
             'SocMed_SharedGen',
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
             'BirthGender',
             'MaritalStatus',
             'SocMed_DiscussHCP',
             'SocMed_TrueFalse',
             'SocMed_SameViews'
            ]

X_model_2 = X_model_1 + ['SocMed_SharedPers', 'SocMed_SharedGen', 'SocMed_WatchedVid']
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

results = []

# 각 모델에 대해 반복문 실행 후 성능 기록
for i, X in enumerate(models, 1):
    # y 변수는 1D 배열로 변환
    X_train, X_test, y_train, y_test = train_test_split(data[X], data[y], test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(4, activation='relu'),
        Dropout(0.5),
        Dense(4, 'softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer = optimizer,
        metrics = ['accuracy']
    )
    # model.summary()

    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        patience = 10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        batch_size = 32,
        epochs = 300,
        shuffle = True,
        verbose = 0,
        validation_split = 0.2,
        callbacks=[early_stopping]
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size = 32)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    results.append(f"Model {i} - {test_accuracy}")
    
    # # 과적합 여부 확인 (훈련/검증 손실 시각화)
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.title(f"Overfitting Test, Model {i}")
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # # plt.show()
    # plt.savefig(f"Overfitting_Test_Model_{i}.png", dpi=300, bbox_inches="tight")
    # plt.close()

print(results)