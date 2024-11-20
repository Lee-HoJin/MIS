import pandas as pd
from scipy.stats import f_oneway
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt   
from sklearn.metrics import confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 데이터 불러오기 Data Loading
data = pd.read_csv("./dataset.csv", sep=",", low_memory=False)

# 데이터 가공 Processing data

variables = ['MisleadingHealthInfo',
             'BirthGender',
             'MaritalStatus',
             'Education',             
             'IncomeFeelings',
             'IncomeRanges',
             'WorkFullTime',
             'NoticeCalorieInfoOnMenu',
             'WearableDevTrackHealth',
             'UsedHealthWellnessApps2',  
             'EthnicGroupBelonging',
             'ConfidentInternetHealth',
             'ConfidentMedForms',
             'GeneralHealth',
             'OwnAbilityTakeCareHealth',
             'SocMed_MakeDecisions', 'SocMed_DiscussHCP', 'SocMed_TrueFalse', 'SocMed_SameViews',
             'SocMed_Visited', 'SocMed_SharedPers', 'SocMed_SharedGen', 'SocMed_Interacted', 'SocMed_WatchedVid']


################# 수치화 Convering to Numeric Value
data['MisleadingHealthInfo'] = data['MisleadingHealthInfo'].replace({
    'I do not use social media': -1,
    'A little': 0,
    'Some': 1,
    'A lot': 2
})

## 여성이 1, 남성이 0 (여성 응답자가 더 많음)
data['BirthGender'] = np.where(data['BirthGender'] == 'Female', 1, 0)

data['MaritalStatus'] = data['MaritalStatus'].replace({
    'Single, never been married' : 0,
    'Separated' : 1,
    'Widowed': 2,
    'Divorced' : 3,
    'Living as married or living with a romantic partner': 4,
    'Married': 5
})

data['Education'] = data['Education'].replace({
    'Less than 8 years' : 0,
    '8 through 11 years' : 1,
    '12 years or completed high school': 2,
    'Post high school training other than college (vocational or ' : 3,
    'Some college': 4,
    'College graduate': 5,    
    'Postgraduate': 6
})

data['IncomeFeelings'] = data['IncomeFeelings'].replace({
    'Living comfortably on present income' : 3,
    'Getting by on present income' : 2,
    'Finding it difficult on present income': 1,
    'Finding it very difficult on present income' : 0,
})

data['IncomeRanges'] = data['IncomeRanges'].replace({
    '$200,000 or more' : 8,
    '$100,000 to $199,999' : 7,
    '$75,000 to $99,999': 6,
    '$50,000 to $74,999' : 5,
    '$35,000 to $49,999' : 4,
    '$20,000 to $34,999' : 3,
    '$15,000 to $19,999' : 2,
    '$10,000 to $14,999' : 1,
    '$0 to $9,999' : 0
})


yes_no_questions = ['WorkFullTime',
                    'NoticeCalorieInfoOnMenu',
                    'WearableDevTrackHealth'
                    ]

for target in yes_no_questions:
    data[target] = data[target].replace({
        'No' : 0,
        'Yes' : 1
    })   
    
data['UsedHealthWellnessApps2'] = data['UsedHealthWellnessApps2'].replace({
    'No' : 0,
    'Yes' : 1,
    'I don\'t have any health apps on my tablet or smartphone' : 0
})

data['EthnicGroupBelonging'] = data['EthnicGroupBelonging'].replace({
    'Strongly disagree' : 0,
    'Disagree' : 1,
    'Neither agree nor disagree': 2,
    'Agree' : 3,
    'Strongly agree' : 4
})

data['ConfidentInternetHealth'] = data['ConfidentInternetHealth'].replace({
    'Not confident at all' : 0,
    'A little confident' : 1,
    'Somewhat confident': 2,
    'Very confident' : 3,
    'Completely confident' : 4
})

B14 = ['SocMed_MakeDecisions', 'SocMed_DiscussHCP', 'SocMed_TrueFalse', 'SocMed_SameViews']
for target in B14 :
    data[target] = data[target].replace({
        'Strongly disagree' : 0,
        'Somewhat disagree' : 1,
        'Somewhat agree' : 2,
        'Strongly agree' : 3,
        'Strongle agree' : 4,
        'Strongle agree' : 5
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
    
# C1
data['FreqGoProvider'] = data['FreqGoProvider'].replace({
    'None' : 0,    
    '1 time': 1,
    '2 times': 2,
    '3 times': 3,
    '4 times': 3,
    '5-9 times': 4,
    '10 or more times': 5,    
})

C3 = ['ChanceAskQuestions', 'FeelingsAddressed', 'InvolvedDecisions', 'UnderstoodNextSteps', 'ExplainedClearly', 'SpentEnoughTime', 'HelpUncertainty']
for target in C3 :
    data[target] = data[target].replace({
        'Never' : 0,
        'Sometimes' : 1,
        'Usually' : 2,
        'Always' : 3,
    })

# C6
data['HealthInsurance2'] = data['HealthInsurance2'].replace({
    'No' : 0,
    'Yes': 1
})

# C7 
data['ConfidentMedForms'] = data['ConfidentMedForms'].replace({
    'Not at all' : 0,
    'A little' : 1,
    'Somewhat': 2,
    'Very' : 3,
})

# C8
data['TrustHCSystem'] = data['TrustHCSystem'].replace({
    'Not at all' : 0,
    'A little' : 1,
    'Somewhat': 2,
    'Very' : 3,
})

# H1
data['GeneralHealth'] = data['GeneralHealth'].replace({
    'Poor' : 0,
    'Fair' : 1,
    'Good': 2,
    'Very good' : 3,
    'Excellent' : 4
})

# E2
data['HCPEncourageOnlineRec2'] = data['HCPEncourageOnlineRec2'].replace({
    'No' : 0,
    'Yes': 1
})

# H2
data['OwnAbilityTakeCareHealth'] = data['OwnAbilityTakeCareHealth'].replace({
    'Not confident at all' : 0,
    'A little confident' : 1,
    'Somewhat confident': 2,
    'Very confident' : 3,
    'Completely confident' : 4
})

# H3
data['UndMedicalStats'] = data['UndMedicalStats'].replace({
    'Very hard' : 0,
    'Hard' : 1,
    'Easy': 2,
    'Very easy' : 3
})

# H5
data['TalkHealthFriends'] = data['TalkHealthFriends'].replace({
    'No' : 0,
    'Yes' : 1
})

H6 = ['MedConditions_Diabetes', 'MedConditions_HighBP', 'MedConditions_HeartCondition', 'MedConditions_LungDisease', 'MedConditions_Depression']
for target in H6 :
    data[target] = data[target].replace({
        'No' : 0,
        'Yes' : 1,
    })
    
K2 = ['HCPShare_FoodIssues', 'HCPShare_TranspIssues', 'HCPShare_HousingIssues']
for target in K2 :
    data[target] = data[target].replace({
        'Very uncomfortable' : 0,
        'Somewhat uncomfortable' : 1,
        'Somewhat comfortable' : 2,
        'Very comfortable' : 3,
    })

# N4  
data['Smoke100'] = data['Smoke100'].replace({
    'No' : 0,
    'Yes' : 1
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

# M3
data['TimesStrengthTraining'] = data['TimesStrengthTraining'].replace({
        'None' : 0,
        '1 day per week' : 1,
        '2 days per week' : 2,
        '3 days per week' : 3,
        '4 days per week' : 4,
        '5 days per week' : 5,
        '6 days per week' : 6,
        '7 days per week' : 7,
    })

data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
data['Weight'] = pd.to_numeric(data['Age'], errors='coerce')
data['AverageSleepNight'] = pd.to_numeric(data['Age'], errors='coerce')

############################    변수 입력
############################ VVVVVVVVVVVVVVVV

y = 'SocMed_MakeDecisions'  # y는 단일 변수이므로 리스트에서 문자열로 변경

X_model_1 = ['Age',
             'IncomeRanges',
             # 'Education',
             'MaritalStatus',
             'BirthGender',
             'SocMed_DiscussHCP',
             # 'MedConditions_Depression',
             'SmokeNow',
            ]
X_model_2 = X_model_1 + B12
X_model_3 = X_model_2 + ['MisleadingHealthInfo']

models = [X_model_1, X_model_2, X_model_3]

# 'Social Media' 사용자들의 응답만 남기기 위함
data['MisleadingHealthInfo'] = data['MisleadingHealthInfo'].replace(-1, np.nan)

for target in ['MisleadingHealthInfo'] + X_model_3 + [y]:  # y를 리스트로 감싸지 않고 단일 변수로 처리
    # target 컬럼에서 숫자가 아닌 값들로 이루어진 행을 제거
    data = data[~data[target].apply(lambda x: isinstance(x, str))]

    # 결측치 제거
    data = data.dropna(subset=[target])
    
    # 고유값 출력
    # print(f"{target} = {data[target].unique()}")
    
    try:
        # 'typecasting' 변수에 대해 int 변환 시도
        data[target] = data[target].astype(int)
    except ValueError as e:
        print(f"Error casting {target}: {e}")
        # 변환 실패한 변수는 건너뜀


########################### Regression Part 회귀 분석 파트

# 성능 결과를 저장할 딕셔너리
model_performance = {
    'Model': [],
    'R-squared': [],
    'MSE': [],
    'F1-score': [],
    'Random Forest Accuracy': [],
}

# 각 모델에 대해 반복문 실행 후 성능 기록
for i, X in enumerate(models, 1):
    # y 변수는 1D 배열로 변환
    X_train, X_test, y_train, y_test = train_test_split(data[X], data[y], test_size=0.2, random_state=42)
    
    # VIF 계산
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_train.columns
    vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    print("Variance Inflation Factor:")
    print(vif_data)

    # 선형 회귀 모델 생성 및 학습
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # 예측
    y_pred = linear_model.predict(X_test)

    # 평가
    mse = mean_squared_error(y_test, y_pred)
    r_squared = linear_model.score(X_test, y_test)
    
    # 학습 데이터에 상수항 추가 (절편)
    X_train_const = sm.add_constant(X_train)

    # 선형 회귀 모델 생성 및 학습
    model = sm.OLS(y_train, X_train_const).fit()

    # 모델 요약 결과 출력
    # print(model.summary())

    # 랜덤 포레스트 모델 훈련
    rf_model = RandomForestClassifier(n_estimators=300,  class_weight='balanced', max_depth=8, min_samples_split=5, random_state=42)

    # 모델 학습
    rf_model.fit(X_train, y_train)

    # 예측
    y_pred_rf = rf_model.predict(X_test)

    # 정확도 평가
    accuracy = accuracy_score(y_test, y_pred_rf)

    # F1-score 계산
    f1 = f1_score(y_test, y_pred_rf, average='micro')
    
    # 교차 검증
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    
    # 각 폴드에서의 성능 점수 출력
    print("Cross-validation scores:", cv_scores)

    # 평균 성능 출력
    print("Mean CV score:", np.mean(cv_scores))

    # 표준편차 출력
    print("Standard deviation of CV scores:", np.std(cv_scores))

    # 성능 기록
    model_performance['Model'].append(f'Model {i}')
    model_performance['R-squared'].append(r_squared)
    model_performance['MSE'].append(mse)
    model_performance['Random Forest Accuracy'].append(accuracy)
    model_performance['F1-score'].append(f1)
    
    
# 성능 결과 출력
performance_df = pd.DataFrame(model_performance)
print(performance_df)

## 잔차 히스토그램
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30, color="blue")
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid()

# 그림 저장
plt.savefig("residuals_histogram.png", dpi=300, bbox_inches="tight")
plt.close()  # plot 창 닫기

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# y_test와 y_pred는 다중 클래스 실제값과 예측값
cm = confusion_matrix(y_test, y_pred)
