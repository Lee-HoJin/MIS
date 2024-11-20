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
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

data = pd.read_csv("./dataset.csv", sep=",", low_memory=False)

# 데이터 가공

variables = ['MisleadingHealthInfo',
             'SexualOrientation',
             'BirthGender',
             'Education',
             'IncomeFeelings',
             'IncomeRanges',
             'WorkFullTime',
             'EthnicGroupBelonging',
             'ConfidentInternetHealth',
             'ConfidentMedForms',
             'NoticeCalorieInfoOnMenu',
             'HealthRecsConflict',
             'HealthRecsChange',
             'UsedHealthWellnessApps2',
             'WearableDevTrackHealth',
             'DiscriminatedMedCare',
             'GeneralHealth',
             'OwnAbilityTakeCareHealth',
             'SocMed_MakeDecisions', 'SocMed_DiscussHCP', 'SocMed_TrueFalse', 'SocMed_SameViews',
             'SocMed_Visited', 'SocMed_SharedPers', 'SocMed_SharedGen', 'SocMed_Interacted', 'SocMed_WatchedVid',
             'ChanceAskQuestions', 'FeelingsAddressed', 'InvolvedDecisions', 'UnderstoodNextSteps', 'ExplainedClearly', 'SpentEnoughTime', 'HelpUncertainty']

data['MisleadingHealthInfo'] = data['MisleadingHealthInfo'].replace({
    'I do not use social media': -1,
    'A little': 0,
    'Some': 0,
    'A lot': 1
})
# valid_values = [0, 1, 2, 3]
# data = data[data['MisleadingHealthInfo'].isin(valid_values)]

# 수치화
data['HealthRecsConflict'] = data['HealthRecsConflict'].replace({
    'Never': 0,
    'Rarely': 0,
    'Often': 1,    
    'Very Often': 1    
})

data['HealthRecsChange'] = data['HealthRecsChange'].replace({
    'Never': 0,
    'Rarely': 0,
    'Often': 1,    
    'Very Often': 1
})

data['UsedHealthWellnessApps2'] = data['UsedHealthWellnessApps2'].replace({
    'No' : 0,
    'Yes' : 1,
    'I don\'t have any health apps on my tablet or smartphone' : 0
})

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
    'Living comfortably on present income' : 1,
    'Getting by on present income' : 1,
    'Finding it difficult on present income': 0,
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

data['EthnicGroupBelonging'] = data['EthnicGroupBelonging'].replace({
    'Strongly disagree' : 0,
    'Disagree' : 0,
    'Neither agree nor disagree': 1,
    'Agree' : 2,
    'Strongly agree' : 2
})

data['ConfidentInternetHealth'] = data['ConfidentInternetHealth'].replace({
    'Not confident at all' : 0,
    'A little confident' : 0,
    'Somewhat confident': 1,
    'Very confident' : 2,
    'Completely confident' : 2
})


yes_no_questions = ['WorkFullTime',
                    'NoticeCalorieInfoOnMenu',
                    'WearableDevTrackHealth',
                    'DiscriminatedMedCare']

for target in yes_no_questions:
    data[target] = data[target].replace({
        'No' : 0,
        'Yes' : 1
    })   

B14 = ['SocMed_MakeDecisions', 'SocMed_DiscussHCP', 'SocMed_TrueFalse', 'SocMed_SameViews']
for target in B14 :
    data[target] = data[target].replace({
        'Strongly disagree' : 0,
        'Somewhat disagree' : 0,
        'Somewhat agree' : 1,
        'Strongly agree' : 1,
        'Strongle agree' : 1,
        'Strongle agree' : 1
    })

B12 = ['SocMed_Visited', 'SocMed_SharedPers', 'SocMed_SharedGen', 'SocMed_Interacted', 'SocMed_WatchedVid']
for target in B12 :
    data[target] = data[target].replace({
        'Never' : 0,
        'Less than once a month' : 1,
        'A few times a month' : 1,
        'At least once a week' : 1,
        'Almost every day' : 2
    })
    
C3 = ['ChanceAskQuestions', 'FeelingsAddressed', 'InvolvedDecisions', 'UnderstoodNextSteps', 'ExplainedClearly', 'SpentEnoughTime', 'HelpUncertainty']
for target in C3 :
    data[target] = data[target].replace({
        'Never' : 0,
        'Sometimes' : 0,
        'Usually' : 1,
        'Always' : 2,
    })

# C6
data['HealthInsurance2'] = data['HealthInsurance2'].replace({
    'No' : 0,
    'Yes': 1
})

# C7 
data['ConfidentMedForms'] = data['ConfidentMedForms'].replace({
    'Not at all' : 0,
    'A little' : 0,
    'Somewhat': 1,
    'Very' : 2,
})

# C8
data['TrustHCSystem'] = data['TrustHCSystem'].replace({
    'Not at all' : 0,
    'A little' : 0,
    'Somewhat': 1,
    'Very' : 2,
})

# H1
data['GeneralHealth'] = data['GeneralHealth'].replace({
    'Poor' : 0,
    'Fair' : 1,
    'Good': 1,
    'Very good' : 2,
    'Excellent' : 2
})

# H2
data['OwnAbilityTakeCareHealth'] = data['OwnAbilityTakeCareHealth'].replace({
    'Not confident at all' : 0,
    'A little confident' : 1,
    'Somewhat confident': 1,
    'Very confident' : 2,
    'Completely confident' : 2
})

# H3
data['UndMedicalStats'] = data['UndMedicalStats'].replace({
    'Very hard' : 0,
    'Hard' : 0,
    'Easy': 1,
    'Very easy' : 1
})

# H4
data['Deaf'] = data['Deaf'].replace({
    'No' : 0,
    'Yes' : 1
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

H10 = ['LifeHasMeaning', 'ClearSenseDir', 'DeepFulfillment', 'LifeHasPurpose']
for target in H10 :
    data[target] = data[target].replace({
        'Not at all' : 0,
        'A little bit' : 0,
        'Somewhat' : 1,
        'Quite a bit' : 2,
        'Very much' : 2
    })

H11 = ['LittleInterest', 'Hopeless', 'Nervous', 'Worrying']
for target in H11 :
    data[target] = data[target].replace({
        'Not at all' : 0,
        'Several days' : 1,
        'More than half the days' : 2,
        'Nearly every day' : 2
    })
    
H12 = ['FeelLeftOut', 'FeelPeopleBarelyKnow', 'FeelIsolated', 'FeelPeopleNotWithMe']
for target in H12 :
    data[target] = data[target].replace({
        'Never' : 0,
        'Rarely' : 0,
        'Sometimes' : 0,
        'Usually' : 1,
        'Always' : 2
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

# J1
data['ClimateChgHarmHealth'] = data['ClimateChgHarmHealth'].replace({
        'Dont know' : 0,
        'Not at all' : 0,
        'A little' : 0,
        'Some' : 1,
        'A lot' : 2
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

############################    변수 입력
############################ VVVVVVVVVVVVVVVV

targets = ['SocMed_SharedPers', 'SocMed_WatchedVid', 'SocMed_SharedGen', 'MisleadingHealthInfo', 'SocMed_SameViews']
y_vars = ['SocMed_MakeDecisions']

# targets = ['IncomeRanges', 'ConfidentInternetHealth']
# y_vars = ['MisleadingHealthInfo', ]

# targets = ['Deaf', 'MedConditions_Diabetes', 'MedConditions_HighBP', 'MedConditions_HeartCondition', 'MedConditions_LungDisease']+ ['IncomeFeelings']
# y_vars = ['WorkFullTime']

# targets = H10 + H11 + H12
# y_vars = ['MedConditions_HeartCondition', 'MedConditions_Depression']

# targets = ['ConfidentInternetHealth', 'WearableDevTrackHealth'] + ['SocMed_Interacted', 'IncomeRanges'] + ['MedConditions_Depression']
# y_vars = ['UsedHealthWellnessApps2']

for target in targets + y_vars : 
    # target 컬럼에서 숫자가 아닌 값들로 이루어진 행을 제거
    data = data[~data[target].apply(lambda x: isinstance(x, str))]
    
    if target == 'MisleadingHealthInfo' :
        data[target] = data[target].replace(-1, np.nan)
    
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

predictors = data[targets]
predictors = sm.add_constant(predictors)

# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data["Variable"] = predictors.columns
vif_data["VIF"] = [variance_inflation_factor(predictors.values, i) for i in range(predictors.shape[1])]

print(vif_data)

########################### Regression Part 회귀 분석 파트

x_var_ordered = data[targets]
x_var_multi_logit = data[targets]

for y_var in y_vars :
    # data = data.dropna(subset = y_var)
    
    X_train, X_test, y_train, y_test = train_test_split(x_var_multi_logit, data[y_var], test_size=0.2, random_state=42)
        
    # 순서형 로지스틱 회귀 Ordered Logistic Regression
    order_logit_model = OrderedModel(
        data[y_var].astype(int),
        x_var_ordered,
        distr='logit'
    )
    order_logit_result = order_logit_model.fit(method='bfgs')
    print(order_logit_result.summary())
        
    # 다중 회귀 Multionmial Logistc Regression
    multi_logit_model = sm.MNLogit(data[y_var], sm.add_constant(x_var_multi_logit))
    multi_logit_result = multi_logit_model.fit()
    print(multi_logit_result.summary())

    # 성능 평가
    print("___ 성능 검증")
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
        print(f"\n{name}\nMcFadden’s Pseudo R-squared: {pseudo_r2}")
        
        if model == order_logit_result:
            X = x_var_ordered
        else :
            X = x_var_multi_logit
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
        rf = RandomForestClassifier(class_weight='balanced', random_state=42)
        rf.fit(X_train, y_train)

        # 예측
        y_pred = rf.predict(X_test)

        # F1-score 계산
        f1 = f1_score(y_test, y_pred, average='micro')
        print(f"F1-score ({y_var}):", f1)
        print("")
    
    ##### 텐서 플로우 활용
    tensor_y = data[y_var]

    # 데이터 스케일링
    from tensorflow.keras.regularizers import l2
    # from sklearn.preprocessing import MinMaxScaler
    
    scaler = StandardScaler()    
    # scaler = MinMaxScaler()
    tensor_x = scaler.fit_transform(x_var_multi_logit.values)

    # 훈련 및 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(tensor_x, tensor_y, test_size=0.2, random_state=42)

    # 텐서플로우 모델 설정
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)),  # L2 정규화 추가
        Dropout(0.2),  # 과적합 방지를 위한 드롭아웃
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),  # L2 정규화 추가
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # 분류 문제의 경우 sigmoid, 회귀 문제라면 'linear' 선택 가능
    ])

    optimizer = Adam(learning_rate=0.0001)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 조기 종료 설정 (성능이 개선되지 않으면 학습 중단)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 모델 학습
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, verbose = 0 , callbacks=[early_stopping])

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
    
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Random Forest Confusion Matrix')
    plt.show()



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

# Plot confusion matrix for RandomForest model



# from sklearn.model_selection import GridSearchCV
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# # Keras 모델을 Scikit-learn과 호환되게 래핑
# def create_model(learning_rate=0.001, dropout_rate=0.3):
#     model = Sequential([
#         Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
#         Dropout(dropout_rate),
#         Dense(32, activation='relu'),
#         Dropout(dropout_rate),
#         Dense(1, activation='sigmoid')
#     ])
#     optimizer = Adam(learning_rate=learning_rate)
#     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=32, verbose=0)

# param_grid = {
#     'learning_rate': [0.001, 0.0001],
#     'dropout_rate': [0.1, 0.2, 0.3, 0.4],
#     'batch_size': [16, 32, 64, 128],
#     'epochs': [10, 20, 50, 100]
# }

# grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
# grid_result = grid.fit(X_train, y_train)

# # Best parameters
# print(f"Best Parameters: {grid_result.best_params_}")