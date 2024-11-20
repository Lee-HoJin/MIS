import pandas as pd
import numpy as np

# 데이터 불러오기
data = pd.read_csv("./dataset.csv", sep=",", low_memory=False)

# 분석할 변수 리스트
variables = ['MisleadingHealthInfo', 'BirthGender', 'MaritalStatus', 'Education', 'IncomeFeelings', 
             'IncomeRanges', 'WorkFullTime', 'NoticeCalorieInfoOnMenu', 'WearableDevTrackHealth', 
             'EthnicGroupBelonging', 'ConfidentInternetHealth', 'ConfidentMedForms', 'GeneralHealth', 
             'OwnAbilityTakeCareHealth', 'SocMed_MakeDecisions', 'SocMed_DiscussHCP', 'SocMed_TrueFalse', 
             'SocMed_SameViews', 'SocMed_Visited', 'SocMed_SharedPers', 'SocMed_SharedGen', 'SocMed_Interacted', 
             'SocMed_WatchedVid']

# 결과를 저장할 리스트
response_data = []

# 각 변수에 대해 응답별 개수 계산
for var in variables:
    if var in data.columns:
        value_counts = data[var].value_counts(dropna=False)  # NaN 값도 포함해서 계산
        for response, count in value_counts.items():
            response_data.append({'variable': var, 'original response': response, 'original response count': count})

# 전처리 (이상치 및 결측치 제거)
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
data['MisleadingHealthInfo'] = data['MisleadingHealthInfo'].replace(-1, np.nan)

for target in ['MisleadingHealthInfo']:  # 전처리할 변수 리스트
    # target 컬럼에서 숫자가 아닌 값들로 이루어진 행을 제거
    data = data[~data[target].apply(lambda x: isinstance(x, str))]

    # 결측치 제거
    data = data.dropna(subset=[target])

# 숫자 변환 (전처리 후에)
for target in ['MisleadingHealthInfo']:  # 변환할 변수 리스트
    try:
        data[target] = data[target].astype(int)
    except ValueError as e:
        print(f"Error casting {target}: {e}")

# recorded response와 recorded response count 계산
recorded_data = []
for var in variables:
    if var in data.columns:
        value_counts = data[var].value_counts(dropna=False)
        for response, count in value_counts.items():
            recorded_data.append({'variable': var, 'recorded response': response, 'recorded response count': count})

# original data와 recorded data 병합
merged_data = pd.DataFrame(response_data)
recorded_df = pd.DataFrame(recorded_data)

# 컬럼명 확인 후, 필요시 컬럼명을 맞춤
merged_data.rename(columns={'original response': 'response', 'original response count': 'response_count'}, inplace=True)
recorded_df.rename(columns={'recorded response': 'response', 'recorded response count': 'response_count'}, inplace=True)

# 두 DataFrame 병합
final_df = pd.merge(merged_data, recorded_df, on=['variable', 'response'], how='left')

# 최종 결과 저장
final_df.to_csv('final_response_counts.csv', index=False)

print("END")