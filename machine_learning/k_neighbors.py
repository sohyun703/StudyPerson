import random



# 도미의 길이 데이터 (cm)
dorados_length = [random.uniform(35, 50) for _ in range(40)]

# 도미의 무게 데이터 (kg)
dorados_weight = [length * 10 + random.uniform(-5, 5) for length in dorados_length]

# 빙어의 길이 데이터 (cm)
trout_length = [random.uniform(20, 30) for _ in range(40)]

# 빙어의 무게 데이터 (kg)
trout_weight = [length * 5 + random.uniform(-5, 5) for length in trout_length]


import matplotlib.pyplot as plt

plt.scatter(dorados_length,dorados_weight)
plt.xlabel('length_dorados')
plt.ylabel('weight_dorados')
plt.show()

print(trout_length,trout_weight)

plt.scatter(trout_length,trout_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

length = dorados_length+trout_length
weight = dorados_weight+trout_weight

#리스트 내포
fish_data = [[l,w] for l,w in zip(length,weight)]



#정답 리스트 -> 어떤 게 답인지 알려주지 않으면 뭐가 뭔지 알 수 없음(데이터만 제공하게 됨) -> 지도 학습 #이진분류는 대부분 1 0으로 함.답은 1로 하면 됨.
fish_target =[1]*40 + [0]*40
print(fish_target)

#k-최근접 이웃 -> 주위의 5개 중 몇개의 데이터가 가장 많이 가까이 있는지.(기본)

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(fish_data,fish_target) #fit(데이터 , 정답) -> 학습됨 kn을 보통 모델이라고 함

kn.score(fish_data,fish_target) #얼마나 잘 맞추는지 확인해보면 됨



kn.predict([[60,600]]) #예측--> 샘플 넣었을 때 2차원 배열로 넣어주었기 때문에 실제로 예측할 때도 해야


kn49 = KNeighborsClassifier(n_neighbors=80) #n_neighbors 몇 개의 데이터를 주변에서 볼 것인가.

kn49.fit(fish_data,fish_target)
print(kn49.score(fish_data,fish_target)) #도미만 제대로 예측하게 됨

print(40/80)