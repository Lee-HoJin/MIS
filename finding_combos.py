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

X = [#'MisleadingHealthInfo',
    'BirthGender',
    'MaritalStatus',
    'Education',             
    'IncomeFeelings',
    'IncomeRanges',
    'WorkFullTime',
    'NoticeCalorieInfoOnMenu',
    'WearableDevTrackHealth',
    'EthnicGroupBelonging',
    'ConfidentInternetHealth',
    'ConfidentMedForms',
    'GeneralHealth',
    'OwnAbilityTakeCareHealth',
    'SocMed_MakeDecisions', 'SocMed_DiscussHCP', 'SocMed_TrueFalse', 'SocMed_SameViews',
    'SocMed_Visited', 'SocMed_SharedPers', 'SocMed_SharedGen', 'SocMed_Interacted', 'SocMed_WatchedVid'
    ]


# 'Social Media' 사용자들의 응답만 남기기 위함
data['MisleadingHealthInfo'] = data['MisleadingHealthInfo'].replace(-1, np.nan)

for target in ['MisleadingHealthInfo'] + X + [y]:  # y를 리스트로 감싸지 않고 단일 변수로 처리
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

dependent_var = y
independent_vars = X  # 독립변수 리스트

# 데이터셋 나누기
import itertools
X = data[independent_vars]
y = data[dependent_var]

# 결과 저장
best_combinations = []
threshold = 0.5  # R-squared 기준

# 모든 가능한 독립변수 조합 탐색
for r in range(1, len(independent_vars) + 1):
    for combo in itertools.combinations(independent_vars, r):
        X_subset = X[list(combo)]
        
        # 데이터 나누기 (학습 & 테스트 세트)
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42)
        
        # 회귀 모델 학습
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 모델 평가 (R-squared)
        r_squared = model.score(X_test, y_test)
        
        # R-squared가 기준을 넘는 경우 저장
        if r_squared > threshold:
            best_combinations.append((combo, r_squared))

# 결과 출력
best_combinations = sorted(best_combinations, key=lambda x: x[1], reverse=True)
for combo, r_squared in best_combinations:
    print(f"Combination: {combo}, R-squared: {r_squared:.4f}")