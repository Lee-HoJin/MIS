import pandas as pd
from scipy.stats import f_oneway
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

data = pd.read_csv("./dataset.csv", sep=",", low_memory=False)

# 데이터 가공
## missing data 제거
variables = ['MisleadingHealthInfo',
             'SexualOrientation',
             'BirthGender',
             'Education',
             'IncomeFeelings',
             'IncomeRanges',
             'WorkFullTime',
             'EthnicGroupBelonging',
             'TotalHousehold',
             'NoticeCalorieInfoOnMenu',
             'HealthRecsConflict',
             'UsedHealthWellnessApps2',
             'WearableDevTrackHealth',
             'SocMed_MakeDecisions', 'SocMed_DiscussHCP', 'SocMed_TrueFalse', 'SocMed_SameViews']


for target in variables: 
    data = data[~data[target].isin(['Missing data (Not Ascertained)', 
                                                'Missing data (Web partial - Question Never Seen)',
                                                'None',
                                                'Multiple responses selected in error',
                                                'Inapplicable, coded 5 in MisleadingHealthInfo',
                                                'Question answered in error (Commission error)'])]
    data[target] = data[target].dropna()

# 수치화
data['MisleadingHealthInfo'] = data['MisleadingHealthInfo'].replace({
    'I do not use social media': 0,
    'A little': 1,
    'Some': 2,
    'A lot': 3
})
# data = data[data['MisleadingHealthInfo'] != -1]

data['HealthRecsConflict'] = data['HealthRecsConflict'].replace({
    'Never': 0,
    'Rarely': 1,
    'Often': 2,    
    'Very Often': 3    
})

data['UsedHealthWellnessApps2'] = data['UsedHealthWellnessApps2'].replace({
    'No' : 0,
    'Yes' : 1,
    'I don\'t have any health apps on my tablet or smartphone' : -1,
    'Inapplicable, coded 1 in HaveDevice_CellPh or coded 1 in Hav' : -1,
    'Question answered in error (Commission Error)' : -1,
    'Missing data (Filter Missing)' : -1
})
data = data[data['UsedHealthWellnessApps2'] != -1]

data['Education'] = data['Education'].replace({
    'Less than 8 years' : 0,
    '8 through 11 years' : 1,
    '12 years or completed high school': 2,
    'Post high school training other than college (vocational or ' : 3,
    'Some college': 4,
    'College graduate': 5,    
    'Postgraduate': 6,
})

data['IncomeFeelings'] = data['IncomeFeelings'].replace({
    'Living comfortably on present income' : 0,
    'Getting by on present income' : 1,
    'Finding it difficult on present income': 2,
    'Finding it very difficult on present income' : 3,
})

data['IncomeRanges'] = data['IncomeRanges'].replace({
    '$200,000 or more' : 0,
    '$100,000 to $199,999' : 1,
    '$75,000 to $99,999': 2,
    '$50,000 to $74,999' : 3,
    '$35,000 to $49,999' : 4,
    '$20,000 to $34,999' : 5,
    '$15,000 to $19,999' : 6,
    '$10,000 to $14,999' : 7,
    '$0 to $9,999' : 8
})

data['EthnicGroupBelonging'] = data['EthnicGroupBelonging'].replace({
    'Strongly disagree' : 0,
    'Disagree' : 1,
    'Neither agree nor disagree': 2,
    'Agree' : 3,
    'Strongly agree' : 4
})

yes_no_questions = ['WorkFullTime',
                    'NoticeCalorieInfoOnMenu',
                    'WearableDevTrackHealth']

for target in yes_no_questions: 
    data[target] = np.where(data[target] == 'Yes', 1, 0)

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
    

#### 이 변수는 순서형 회귀 분석에 사용되면 안 됨!
data['SexualOrientation_numeric'] = data['SexualOrientation'].replace({
    'Heterosexual, or straight (Skip OS variable)' : 0,
    'Bisexual (Skip OS variable)' : 1,
    'Homosexual, or gay or lesbian (Skip OS variable)': 2,
    'Something else - Specify' : 3
})

# 성적 지향성 각각 더미 변수 생성: 각각의 값에 대해 0 또는 1로 표시
data['Heterosexual'] = (data['SexualOrientation'] == 'Heterosexual, or straight (Skip OS variable)').astype(int)
data['Bisexual'] = (data['SexualOrientation'] == 'Bisexual (Skip OS variable)').astype(int)
data['Homosexual'] = (data['SexualOrientation'] == 'Homosexual, or gay or lesbian (Skip OS variable)').astype(int)
data['SomethingElse'] = (data['SexualOrientation'] == 'Something else - Specify').astype(int)
SexOri_dummies = ['Heterosexual', 'Bisexual', 'Homosexual', 'SomethingElse']

## 이성애자(heterosexual)이면 1, 그 외 모두 0
data['SexualOrientation_binary'] = np.where(data['SexualOrientation'] == 'Heterosexual, or straight (Skip OS variable)', 1, 0)

## 여성이 1, 남성이 0 (여성 응답자가 더 많음)
data['BirthGender'] = np.where(data['BirthGender'] == 'Female', 1, 0)

data['SocialMedia_Binary'] = np.where(data['MisleadingHealthInfo'] == 0, 0, 1)

# for VIF
predictors = data[['SocMed_MakeDecisions', 'SocMed_TrueFalse']]
predictors = sm.add_constant(predictors)

# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data["Variable"] = predictors.columns
vif_data["VIF"] = [variance_inflation_factor(predictors.values, i) for i in range(predictors.shape[1])]

print(vif_data)

# Regression Part 회귀 분석 파트
data = data.dropna(subset=['MisleadingHealthInfo'])

data['IncomeRanges_sqr'] = data['IncomeRanges'] ** 2

# X = data[['UsedHealthWellnessApps2', 'WearableDevTrackHealth', 'IncomeFeelings', 'IncomeRanges', 'WorkFullTime', 'NoticeCalorieInfoOnMenu']]
# x_var_multi_logit = data[['UsedHealthWellnessApps2', 'WearableDevTrackHealth', 'BirthGender', 'IncomeFeelings', 'WorkFullTime', 'Education']]

X = data[['SocMed_MakeDecisions', 'MisleadingHealthInfo']]
x_var_multi_logit = data[['SocMed_MakeDecisions', 'IncomeRanges']]
y_vars = ['HealthRecsConflict']

for y_var in y_vars :
    # 순서형 로지스틱 회귀 Ordered Logistic Regression
    order_logit_model = OrderedModel(
        data[y_var].dropna(),
        X,
        distr='logit'
    )
    order_logit_result = order_logit_model.fit(method='bfgs')
    print(order_logit_result.summary())
    
    # 다중 회귀 Multionmial Logistc Regression
    multi_logit_model = sm.MNLogit(data[y_var], sm.add_constant(x_var_multi_logit))
    multi_logit_result = multi_logit_model.fit()
    # print(multi_logit_result.summary())

    # 성능 평가
    print("___성능 검증")
    models = [order_logit_result, multi_logit_result]
    for model in models :
        llf = model.llf  # 학습된 모델의 로그 우도
        llnull = model.llnull  # 상수항만 포함된 모델의 로그 우도

        # McFadden의 Pseudo R-squared 계산
        pseudo_r2 = 1 - (llf / llnull)
        
        if model == order_logit_result :
            name = 'Ordered'
        else :
            name = 'Multinomial'
        print(f"McFadden’s Pseudo R-squared of {name}: {pseudo_r2}")
        
    y = data[y_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 모델 학습
    rf_model.fit(X_train, y_train)

    # 예측
    y_pred = rf_model.predict(X_test)

    # 정확도 평가
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy ({y_var}):", accuracy)

    # 랜덤 포레스트 모델 훈련
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # 예측
    y_pred = rf.predict(X_test)

    # F1-score 계산
    f1 = f1_score(y_test, y_pred, average='micro')
    print(f"F1-score ({y_var}):", f1)
    print("")

control_model = OrderedModel(
    data['MisleadingHealthInfo'],
    data[['Education', 'IncomeFeelings', 'IncomeRanges']],
    distr='logit'
)
result_control = control_model.fit(method='bfgs')

# print(result.summary())
# print(result_control.summary())


#############################################################################################

# print(data['Education'].skew())
# print("After the logarithm transformation")
# data['Education_log'] = np.log1p(data['Education'])
# print(data['Education_log'].skew())

# 원하는 컬럼만 히스토그램으로 그리기
# columns_to_plot = ['SexualOrientation', 'Education', 'IncomeFeelings', 'IncomeRanges']

## 각 변수에 대해 Q-Q 플롯 그리기
# for column in columns_to_plot:
#     sm.qqplot(np.array(data[column]), line ='45')
#     plt.title(f'Q-Q Plot for {column}')
#     plt.xlabel(column)
#     plt.show()

## 각 변수에 대해 히스토그램 그리기
# for column in columns_to_plot:
#     plt.hist(data[column], bins=10)
#     plt.title(f'Histogram of {column}')
#     plt.xlabel(column)
#     plt.ylabel('Frequency')
#     plt.show()

## Multinomial Logistic Regression (다중 범주 로지스틱 회귀)
# model = sm.MNLogit(data['MisleadingHealthInfo'], sm.add_constant(data[['IncomeRanges',  'WorkFullTime', 'NoticeCalorieInfoOnMenu']]))
# result = model.fit()

# # 결과 출력
# print(result.summary())


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터 준비
# X = data[['IncomeRanges', 'IncomeFeelings', 'WorkFullTime', 'NoticeCalorieInfoOnMenu']].values
tensor_X = X.values
y = data['HealthRecsConflict'].values

# 데이터 스케일링
scaler = StandardScaler()
X = scaler.fit_transform(tensor_X)

# 훈련 및 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(tensor_X, y, test_size=0.2, random_state=42)

# 텐서플로우 모델 설정
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),  # 과적합 방지를 위한 드롭아웃
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # 분류 문제의 경우 sigmoid, 회귀 문제라면 'linear' 선택 가능
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 조기 종료 설정 (성능이 개선되지 않으면 학습 중단)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 모델 학습
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping])

# 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# 과적합 여부 확인 (훈련/검증 손실 시각화)
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# from statsmodels.multivariate.manova import MANOVA

# # 종속 변수들
# Y = data[['SocMed_MakeDecisions', 'SocMed_DiscussHCP', 'SocMed_TrueFalse', 'SocMed_SameViews']]

# # 독립 변수
# X = data[['MisleadingHealthInfo']]
# X = sm.add_constant(X)

# # 다변량 선형 회귀
# manova = MANOVA(endog=Y, exog=X)
# print(manova.mv_test())

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# # 독립 변수와 종속 변수
# X = data[['MisleadingHealthInfo']]
# Y = data[['SocMed_MakeDecisions', 'SocMed_DiscussHCP', 'SocMed_TrueFalse', 'SocMed_SameViews']]

# # 모델 생성 및 학습
# model = LinearRegression()
# model.fit(X, Y)

# # 예측
# y_pred = model.predict(X)

# # 성능 평가
# print(f"Mean Squared Error: {mean_squared_error(Y, y_pred)}")