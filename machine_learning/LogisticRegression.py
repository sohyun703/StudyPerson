from google.colab import files
import io
import pandas as pd
import numpy as np

uploaded = files.upload()

fish = pd.read_csv(io.BytesIO(uploaded['fish.csv']))
fish.info()

#fish 데이터프레임의 상위 5개를 확인

fish.head()

#어떤 종류의 생선이 있는지 Species 열에서 고유한 값을 추출한다.

print(pd.unique(fish['Species']))

1.2 입력데이터와 정답데이터 만들기


*   Species 열을 타깃(정답)데이터로 만들고 나머지 5개 열은 입력 데이터로 만든다.



fish_input = fish[['Weight',"Length","Diagonal",	"Height"	,"Width"]].to_numpy()
fish_target= fish['Species'].to_numpy()

데이터 준비 -> 표준화 전처리

1.3 훈련세트와 테스트 세트 만들기

from sklearn.model_selection import train_test_split

train_input,test_input,train_target,test_target = train_test_split(fish_input,fish_target,random_state=42)


1.4 표준화 전처리

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input) #표준화값 구하기

#훈련세트와 테스트 세트를 각각 변형된 값으로 변환시켜준다. -> 변환기
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

## 2. K 최근접 이웃 분류 모델의 확률 예측
2.1 모델 객체를 생성하고 훈련과 평가를 한다.

#k-최근접 이웃 분류 클래스를 임포트

from sklearn.neighbors import KNeighborsClassifier

#최근접 이웃 개수를 3으로 지정하여 모델 객체를 만든다.

kn = KNeighborsClassifier(n_neighbors=3)

#전처리된 훈련 세트로 모델을 훈련한다.

kn.fit(train_scaled,train_target)

#훈련세트와 테스트세트 평가

print(kn.score(train_scaled,train_target))
kn.score(test_scaled,test_target)

2.2 타깃값의 종류(7개)들을 확인하고, 타깃값을 예측해본다.

#KneighborsClassifier에서 정렬된 타깃값은 classes_ 속성에 알파벳 순서로 저장되어 있다.

print(kn.classes_)

#predict로 타깃값을 예측한다.
#테스트 세트에 있는 처음 5개의 ㄱ샘플의 타깃값을 예측해보자.

print(kn.predict(test_scaled[:5]))

2.3 샘플 5개에 대한 예측은 어떤 확률로 만들어졌을까?

import numpy as np

#사이킷런의 분류모델 predict_proba()로 클래스별 확률값을 반환


proba = kn.predict_proba(test_scaled[:5])

#decimals은 반올림해서 소숫점 4번째자리까지 표시한다.

print(np.round(proba,decimals=4))

#인덱스 3번째 샘플의 최근접 이웃의 클래스를 확인해보자
#n_neighbors=3으로 모델을 생성했기 때문에 이웃의 숫자는 3개이다.

distances, indexes = kn.kneighbors(test_scaled[3:4])
print(distances)
print(indexes)
print(train_target[indexes])

##4. 로지스틱 회귀로 이진 분류 수행하기


*   도미와 빙어 4개를 사용해서 이진 분류를 수행해보자



from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(train_input,train_target)


print(lr.predict(test_input[:5]))
print(lr.predict_proba(test_input[:5]))

char_arr =np.array(['a','f','b','c','d','e'])
print(char_arr[[True,True,False,False,False,False]])

print(train_target[:5])
print(train_scaled[:5])

#train_target 배열에서 도미와 빙어일 경우 True 나머지는 False를 사용해서 도미 빙어 배열을 만들어준다.
bream_smelt_indexes = (train_target=="Bream") | (train_target=='Smelt')

#bream_smelt_indexes배열을 이용해서 훈련세트에서 도미와 빙어 데이터만 골라낸다. #훈련한 데이터에 대해서 가지고 옴
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

4.2 로지스틱 회귀 모델을 훈련시킨다.

#사이킷런에서 로지스틱 회귀 모델을 임포트한다.

from sklearn.linear_model import LogisticRegression

#로지스틱 회귀 모델 객체를 만들고 훈련시킨다.

lr = LogisticRegression()
lr.fit(train_bream_smelt,target_bream_smelt)

4.3 샘플을 예측하고 예측확률을 확인한다.


#훈련한 모델을 사용해 train_bream_smelt에 있는 5개의 샘플을 예측해보자

print(lr.predict(train_bream_smelt[:5]))

print(train_bream_smelt[:5])

#predict_proba는 예측확률 제공
#처음 음성확률 이후 양성확률
print(lr.predict_proba(train_bream_smelt[:5]))

print(lr.classes_)

4.4 로지스틱 회귀 계수를 확인하고 학습한 방정식을 구하자.

print(lr.coef_)
print(lr.intercept_)

#decision_function()으로 z값을 출력할 수 있다.

decision = lr.decision_function(train_bream_smelt[:5])
print(decision)

#z값을 시그모이드 함수에 통과하면 확률을 구할 수 있고 파이썬의 scipy 라이브러리에 시그모이드 함수가 존재한다.
from scipy.special import expit

#양성 클래스에 대한 z 값을 반환하기 때문에 음성 클래스는 1에서 빼주면 된다.
print(expit(decision))

## 5 로지스틱 회귀로 다중 분류 수행하기

5.1 로지스틱 회귀 모델 객체를 만들고 훈련 및 평가하자

#규제를 완화하기 위해 c=20(c는 알파와 달리 작을수록 규제가 완화됨), 충분한 반복횟수 1000을 지정해주어 로지스틱 회귀 모델 객체를 만들어준다.

lr = LogisticRegression(C=20,max_iter=1000)

#훈련 시키기

lr.fit(train_scaled,train_target)

#훈련 세트와 테스트 세트 평가

print(lr.score(train_scaled,train_target))
print(lr.score(test_scaled,test_target))

5.2 예측과 예측 확률을 출력한다.

print(lr.predict(test_scaled[:5]))


proba =lr.predict_proba(test_scaled[:5])

print(np.round(proba,decimals=3))

#생선의 종류를 classes_ 속성으로 확인한다.

print(lr.classes_)

#다중 분류의 경우 선형 방정식의 계수와 절편을 확인한다.

print(lr.coef_.shape,lr.intercept_.shape)
print(lr.coef_)

decision = lr.decision_function(test_scaled[:5])
print(np.round(decision,decimals=2))

print(lr.predict(test_scaled[:5]))

#이진 분류는 시그모어 함수, 다중 분류는 소프트맥스 함수

from scipy.special import softmax

proba = softmax(decision,axis=1)
print(np.round(proba,decimals=3))

proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba,decimals=3))