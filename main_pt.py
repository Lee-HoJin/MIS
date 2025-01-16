import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
torch.cuda.init()  # CUDA 초기화
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

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

##################################################

class CustomDataset(Dataset) :
    def __init__(self, X, y) :
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32) 

    def __len__(self) :
        return len(self.X)
    
    def __getitem__(self, idx) :
        return self.X[idx], self.y[idx]

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

###################################################
########## 회귀 분석 파트 Regression Part ##########
###################################################

class EarlyStopping :
    def __init__(self, patience = 5, delta = 0) :
        self.patience = patience # 개선되지 않은 epoch 수
        self.delta = delta # 개선 기준 (손실 감소량)
        self.best_loss = None
        self.counter = 0 # 개선되지 않은 epoch 수
        self.early_stop = False

    def __call__(self, val_loss, model) :
        if self.best_loss is None :
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta :
            self.best_loss = val_loss
            self.counter = 0 # 성능이 개선되었으므로 카운터 리셋
        else :
            self.counter += 1

        if self.counter >= self.patience :
            self.early_stop = True
            # print(f"Early stopping triggered after {self.patience} epochs without improvement.")

        return self.early_stop

model1_accuracy_sum = 0
model2_accuracy_sum = 0
model3_accuracy_sum = 0

num_of_tests = 10

for iteration in range(num_of_tests) :
    # 각 모델에 대해 반복문 실행 후 성능 기록
    for i, X in enumerate(models, 1):
        print(f"__Model {i}  iteration {iteration + 1}")

        # # 정규화(or Scaling)
        # scaler = StandardScaler()
        # scaler.fit(data[X])
        # X_scaled = scaler.transform(data[X])

        # 데이터셋 생성
        dataset = CustomDataset(data[X], data[y])

        # 데이터셋을 훈련/검증/테스트 세트로 분리
        train_size = int(0.8 * len(dataset))  # 80% 훈련 데이터
        val_size = int(0.1 * len(dataset))  # 10% 검증 데이터
        test_size = len(dataset) - train_size - val_size  # 나머지 10% 테스트 데이터

        # 데이터셋을 훈련, 검증, 테스트로 분리
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        # 데이터 로더 생성
        batch_size = 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for batch in train_loader:
            X_batch, y_batch = batch
            input_features = X_batch.shape[1]
            break

        class NeuralNetwork(nn.Module):
            def __init__(self):
                super().__init__() # 부모 클래스 초기화 메서드를 호출
                self.flatten = nn.Flatten() # 보통 첫 번째 차원은 유지하고 나머지 차원을 모두 곱해서 2차원 텐서로 만듦
                self.linear_relu_stack = nn.Sequential(
                    nn.Linear(input_features, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 4),
                )

            def forward(self, x):
                x = self.flatten(x)
                logits = self.linear_relu_stack(x)
                return logits
            
        def init_weights(m) :
            if isinstance(m, nn.Linear) :
                # He 초기화
                nn.init.kaiming_normal_(m.weight, nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
               
        # 모델 정의 및 훈련
        model = NeuralNetwork()
        # 가중치 초기화 적용
        model.apply(init_weights)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)

        # EarlyStopping 객체 생성
        early_stopping = EarlyStopping(patience = 40, delta = 0.05)

        # 훈련 루프
        num_epochs = 161  # 에폭 수
        for epoch in range(num_epochs):
            model.train()  # 모델을 훈련 모드로 설정
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch in train_loader:
                X_batch, y_batch = batch
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # 옵티마이저 기울기 초기화
                optimizer.zero_grad()

                # 예측값 계산
                outputs = model(X_batch)

                # 손실 계산
                loss = criterion(outputs, y_batch.long())  # CrossEntropyLoss는 정수형 레이블을 사용
                loss.backward()  # 역전파

                # 파라미터 업데이트
                optimizer.step()

                # 손실 추적
                running_loss += loss.item()

                # 정확도 계산
                _, predicted = torch.max(outputs, 1)  # 가장 높은 확률의 클래스 예측
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            if epoch % 20 == 0 :
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

            # 검증 정확도 계산 (매 에폭마다 검증 데이터로 성능 평가)
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    X_batch, y_batch = batch
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch.long())
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    val_total += y_batch.size(0)
                    val_correct += (predicted == y_batch).sum().item()

            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total
            if epoch % 20 == 0:
                print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

            # EarlyStopping 체크
            if early_stopping(val_loss, model):
                print(f"Stopping early at epoch {epoch+1}")
                break
            
        # 마지막 테스트 정확도 계산
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                X_batch, y_batch = batch
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.long())
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                test_total += y_batch.size(0)
                test_correct += (predicted == y_batch).sum().item()

        test_acc = 100 * test_correct / test_total
        test_loss /= len(test_loader)  # 평균 손실
    
        if i == 1 :
            model1_accuracy_sum += test_acc
        elif i == 2 :
            model2_accuracy_sum += test_acc
        elif i == 3:
            model3_accuracy_sum += test_acc

    print("")

# 전체 테스트 반복 결과의 평균
model1_avg_accuracy = model1_accuracy_sum / num_of_tests
model2_avg_accuracy = model2_accuracy_sum / num_of_tests
model3_avg_accuracy = model3_accuracy_sum / num_of_tests

print(f"\n{num_of_tests} processes was conducted")
print(f"Model 1 Average Accuracy: {model1_avg_accuracy:.4f}")
print(f"Model 2 Average Accuracy: {model2_avg_accuracy:.4f}")
print(f"Model 3 Average Accuracy: {model3_avg_accuracy:.4f}")