다항회귀 : 항이 여러개

다중 회귀 : 여러 개의 특성을 이용해 특성을 찾는다. > 다중 회귀, 특성이 2개인 다중회귀 모델은 평면을 학습한다.


**특성공학**

기존의 특성을 사용해 새로운 특성을 뽑아내는 작업
-> 농어길이 *농어길이 -> 새로운 트성을 만들어냄

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

from google.colab import files
import io
import pandas as pd
import numpy as np

uploaded= files.upload()

#read_csv()으로 업로드된 파일을 Byte 단위로 읽어서 데이터 프레임을 생성한다.

df = pd.read_csv(io.BytesIO(uploaded['perch_full (1).csv']))

#to_numpy()를 이용해 넘파이 배열로 변환한다.
perch_full = df.to_numpy()

#shape 속성으로 배열의 크기를 확인한다.

print(perch_full.shape)
print(perch_full)

**2. 변환기**

사이킷런은 **전처리를 하거나 특성을 만드는 클래스**를 변환기 클래스라고 한다.

**fit() -> 새롭게 만들 특성 조합**을 찾기

**transform() -> 실제로 데이터를 변환**

ex) PolynomialFeatures 클래스는 각 특성의 제곱항과 특성끼리 곱한항을 추가해준다.

poly.fit([[2,3]])
poly.transform([[2,3]]) -> 1,2,3,4,6,9
include_bias =>를 통해 절편에 해당하는 1 생략 가능
get_feature_names()를 통해 특성이 어떤 조합으로 만들어졌는지 확인 가능

# 2. 사이킷런의 변환기

## 2.1 다중 특성 만들기

#sklearn.preprocessing 패키지에 포함되어 있는 PolynomialFeautres 클래스를 임포트한다.

from sklearn.preprocessing import PolynomialFeatures

#PolynomialFeatures 를 객체로 만든다.

poly = PolynomialFeatures()
poly.fit([[2,3]])
print(poly.transform([[2,3]]))


#y절편 생략하기

poly = PolynomialFeatures(include_bias=False)
poly.fit([[2,3]])
print(poly.transform([[2,3]]))

## 2.2 농어 데이터의 다중 특성 만들기

from sklearn.model_selection import train_test_split
train_input, test_input = train_test_split(perch_full, random_state=42)

#PolynomialFeaures 클래스의 객체를 생성한다.

poly = PolynomialFeatures(include_bias=False)

poly.fit(train_input) #특성 
train_poly = poly.transform(train_input)

print(train_poly.shape)

poly.get_feature_names_out()

test_poly = poly.transform(test_input)

print(test_poly.shape)

# 3. 다중 회귀 모델 훈련하기


# 다중 회귀 모델을 훈련하는 것은 선형 회귀 모델을 훈련하는 것과 같다.

from sklearn.linear_model import LinearRegression

#변환(추가)된 다중 특성을 사용하여 다중 회귀 모델을 훈련한다.

lr = LinearRegression()
lr.fit(train_poly,train_target)

#훈련 세트에 대한 평가
print(lr.score(train_poly,train_target))

#테스트 세트에 대한 평가

print(lr.score(test_poly,test_target))

# 4. 더 많은 특성 만들기

#PolynomialFeautres 클래스의 degress매개변수를 사용하여 고차항의 최대차수를 지정할 수 있다.

poly = PolynomialFeatures(degree=5,include_bias=False) #최대차항 5, y 절편 없애기

#후ㅜㄴ련 세트에 5제곱 다중 특성이 추가된다.

#fit는 새롭게 만들 특성 조합을 찾고, transform는 실제로 특성으로 변환해준다.

poly.fit(train_input)
train_poly= poly.transform(train_input)
test_poly = poly.transform(test_input)

print(train_poly.shape) #배열의 열의 개수가 추가된 특성의 개수

lr.fit(train_poly,train_target)

print(lr.score(train_poly,train_target))

print(lr.score(test_poly,test_target))

### 규제


*   머신러닝 모델이 너무 과도하게 학습하지 못하도록 훼방하는 것
*  선형회귀모델의 경우 특성에 곱해지는 계수의 크기를 작게 만드는 것이다.
*   규제 전에 표준화
*   추가된 특성의 스케일이 정규화되지 않으면 곱해지는 계수 값도 차이가 나게 된다.






### 6. 규제 전 표준화 -> 계수 차이가 나지 않기 위하여

#규제 전에 추가된 특성들에 대해 표준화를 수행한다.
#사이킷런의 StandardScaler 클래스를 임포트한다.

from sklearn.preprocessing import StandardScaler

#StandaredScaledr 클래스의 객체를 생성한다.

ss = StandardScaler()

#fit와 transform()에 의해 표준화를 수행한다.
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# 7. 릿지 회귀

#릿지 클래스를 임포트한다.

from sklearn.linear_model import Ridge

#릿지 회귀 모델 객체를 만든다.

ridge = Ridge()

#훈련 세트로 릿지회귀모델을 훈련시킨다.

ridge.fit(train_scaled,train_target)
ridge.score(train_scaled,train_target)


#훈련세트로 릿지회귀모델을 평가한다.

print(ridge.score(test_scaled,test_target))

8. 규제의 양 조절

import matplotlib.pyplot as plt

#alpha 값을 바꿀 때마다 score() 결과를 저장할 리스트이다.

train_score= []
test_score=[]

alpha_list=[0.001,0.01,0.1,1,10,100]

for a in alpha_list:
  #알파 값에 따른 릿지 모델 만들기
  ridgd= Ridge(alpha=a)
  #릿지 모델 훈련
  ridge.fit(train_scaled,train_target)
  #훈련 점수와 테스트 점수 저장
  train_score.append(ridge.score(train_scaled,train_target))
  test_score.append(ridge.score(test_scaled,test_target))



plt.plot(np.log10(alpha_list),train_score)
plt.plot(np.log10(alpha_list),test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

ridge=Ridge(alpha=0.1)
ridge.fit(train_scaled,train_target)

print(ridge.score(train_scaled,train_target))
print(ridge.score(test_scaled,test_target))