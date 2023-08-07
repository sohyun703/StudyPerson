# 도미와 빙어의 길이 입력데이터 리스트이다.
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
# 도미와 빙어의 무게 입력데이터 리스트이다.            
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

데이터 넘파이로 정답 데이터와 입력 데이터로 구분하기

import numpy as np

fish_input = np.column_stack((fish_length,fish_weight))

fish_target= np.concatenate((np.ones(35),np.zeros(14)))

from sklearn.model_selection import train_test_split

train_input,test_input,train_target,test_target= train_test_split(fish_input,fish_target,stratify = fish_target,random_state=42)

데이터 전처리 과정

표준 점수로 바꿔주기

표준 점수 = (점수 - 평균)/ 표준 편차

평균 = np.mean()

표준편차 = np.std()




ad = np.mean(fish_input)
std = np.std(fish_input)

#넘파이는 브로드캐스트 기능 제공

scaled_train=(train_input - ad)/std
scaled_test= (test_input-ad)/std

from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(scaled_train,train_target)
kn.score(scaled_test,test_target)



회귀 regression -> 임의의 타깃을 맞춰라
이웃 샘플의 수치를 확인하여 해당 수치들을 활용해 새로운 샘플을 수치를 예측하는 것으로 평균값을 구한다.

import numpy as np
# 특성데이터인 농어의 길이를 넘파이 배열로 생성한다.
perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )
# 타깃데이터인 농어의 무게를 넘파이 배열로 생성한다.
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
     1000.0, 1000.0]
     )

from sklearn.model_selection import train_test_split

# 사이킷런의 train_test_split()를 사용해 훈련세트와 테스트세트로 나눈다.
# 동일한 결과를 얻기위해 랜덤시드 random_state=42 지정한다.
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42) #그 전에는 골고루 나누라고 타깃값에 stratify = fish_target을 넣어주었지만 일반적으로 회귀문제에는 stratify를 지정해주지 않는다.

# 훈련 세트와 테스트 세트를 2차원 배열로 바꾼다.
# 먼저 1열을 지정하고, 행을 -1로 지정하면 행의 개수가 자동 계산된다.
train_input = train_input.reshape(-1, 1) #열이 하나인 2차원 배열 만들어짐
test_input = test_input.reshape(-1, 1)

from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()
knr.fit(train_input,train_target)

knr.score(test_input,test_target)

#mean_absolute_error 패키지는 타깃과 예측의 절대값이 얼마나 차이가 나는지 반환해준다.

from sklearn.metrics import mean_absolute_error
test_predicition = knr.predict(test_input)
mae = mean_absolute_error(test_target,test_predicition)
print(mae)

###과대적합 vs 과소적합

과소적합 : test>train --> 트레인이 부족햇음

과대적합 : train> test --> 트레인을 너무 과하게 했다.

knr.score(test_input,test_target)

knr.score(train_input,train_target)
#과소적합 상탠



만약 과소적합이라면 모델을 조금 더 복잡하게 만들어주면 된다.

k의 수가 늘어나면 늘어날수록 과소적합 상태
k의 수가 적어지면 적어질수록 과대적합

현재는 과소적합 상태 이므로 k의 수를 줄여주면 된다.

knr.n_neighbors= 3
knr.fit(train_input,train_target)

knr.score(train_input,train_target)

knr.score(test_input,test_target)