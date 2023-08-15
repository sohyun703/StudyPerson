## 5장 선형 회귀

### 1. k-최근접 이웃의 한계


1.1 넘파이 배열로 데이터 준비하기

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

1.2 훈련세트와 테스트세트 준비하기

from sklearn.model_selection import train_test_split

# 사이킷런의 train_test_split()를 사용해 훈련세트와 테스트세트로 나눈다.
# 동일한 결과를 얻기위해 랜덤시드 random_state=42 지정한다.
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)

# 훈련 세트와 테스트 세트를 2차원 배열로 바꾼다.
# 먼저 1열을 지정하고, 행을 -1로 지정하면 행의 개수가 자동 계산된다.
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

1.3 회귀 모델의 훈련과 예측하기


*   잘못된 예측 발생

from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors=3)

knr.fit(train_input,train_target)

print(knr.predict([[50]]))

import matplotlib.pyplot as plt

distances,indexes = knr.kneighbors([[50]])
plt.scatter(train_input,train_target)
plt.scatter(train_input[indexes],train_target[indexes],marker='D')
plt.scatter(50,1033,marker='^')
plt.show()

print(np.mean(train_target[indexes]))

print(knr.predict([[100]]))

#100cm 농어의 가장 가까운 이웃까지의 거리와 인덱스를 얻는다.
distances,indexes = knr.kneighbors([[100]])
plt.scatter(train_input,train_target)
plt.scatter(train_input[indexes],train_target[indexes],marker='D')
plt.scatter(100,1033,marker='^')
plt.show()

from sklearn.linear_model import LinearRegression


lr = LinearRegression()
lr.fit(train_input,train_target)
print(lr.score(train_input,train_target))

print(lr.predict([[50]]))

print(lr.coef_,lr.intercept_)


plt.scatter(train_input,train_target)
plt.plot([15,50],[lr.coef_*15+lr.intercept_,lr.coef_*50+lr.intercept_])

plt.scatter(50,1241.8,marker='^')
plt.show()

train_poly = np.column_stack((train_input**2,train_input))
test_poly= np.column_stack((test_input **2,test_input))

print(train_poly.shape,test_poly.shape)

lr.fit(train_poly,train_target)

print(lr.predict([[50**2,50]]))

print(lr.coef_,lr.intercept_)

point = np.arange(15,50)
plt.scatter(train_input,train_target)

plt.plot(point,1.01*point**2-21.6*point +116.05)

plt.scatter([50],[1574],marker='^')
plt.show()