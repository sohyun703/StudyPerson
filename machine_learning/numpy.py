import numpy as np

data= np.array([[1.5,-0.1,3],[0,-3,6.5]])

data

data*10

data+data

data.shape

data.dtype

data1 = [6,7.5,8,0,1]

arr1 = np.array(data1)

arr1

data2 = [[1,2,3,4],[5,6,7,8]]

arr2=np.array(data2)

arr2

arr2.ndim

arr2.shape

arr1.dtype

arr2.dtype

arr2.dtype

np.zeros(10)

np.zeros([3,6])

np.empty((2,3,2))

np.arange(15)

np.ones(10)

arr1 = np.array([1,2,3],dtype=np.float64)

arr2= np.array([1,2,3],dtype=np.int32)

arr1.dtype

arr2.dtype

arr = np.array([1,2,3,4,5])

arr.dtype

float_arr = arr.astype(np.float64)

float_arr.dtype

float_arr

arr = np.array([3.7,-1,2,-3.6,0.5,10.1])

arr

arr.astype(np.int32)

arr

#숫자 형식의 문자열을 담고 있는 배열이 있다면 astype을 사용해 바로 숫자로 변환 가능

numeric_strings = np.array(["1.25","9.6","42"],dtype=np.string_)

numeric_strings.astype(float)

#만일 문자열이 float64로 변환되지 않는 경우 같은 이유로 형 변환이 실패하면 valueError가 발생.

int_array = np.arange(10)

calobers= np.array([.22,.270,.357,.123,.44,.33,.22],dtype = np.float64)

int_array.astype(calobers.dtype) #calobers의 dtype으로 형변환 됨

zeors_unit32 = np.zeros(8,dtype="u4")

zeors_unit32

#4.1.3넘파이 배열의 산술 연산

arr = np.array([[1.,2.,3.],[4.,5.,6.]])

arr

arr*arr

arr-arr

#스칼라 인수 -> 방향은 없지만, 실수 공간에서 크기를 나타냄
#벡터 -> n 차원 공간에서 방향과 크기를 갖음.

#스칼라 인수가 포함된 산술 연산의 경우 배열 내의 모든 원소에 스칼라 인수가 적용된다.

1/arr

arr**2

#크기가 동일한 배열 간의 비교 연산은 불리언 배열 반환

arr2 = np.array([[0.,4.,1.],[7.,2.,12.]])

arr2

arr2>arr

### 크기가 다른 배열간의 연산은 브로드 캐스팅이라고 부른다.

### 4.1.4 색인과 슬라이싱 기초

arr= np.arange(10)

arr

arr[5]

arr[5:8]

arr[5:8] =12

arr

arr_slice = arr[5:8]

arr_slice

arr_slice[1]

arr_slice[1]=12345

arr

arr_slice[:]=64

arr

###  넘파이는 데이터가 복사가 되는 것이 아니라 원본이 바뀐다.
### 복사본이 가지고 싶다면 .copy를 이용하도록 하자

### 2차원 배열에서 각 색인에 해당하는 원소는 스칼라 값이 아닌 1차원 배열

arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])

arr2d[2]

### 아래 두 표현은 동일하다.

arr2d[0][2]

arr2d[0,2]

arr3d = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])

arr3d

arr3d[0]

old_values = arr3d[0].copy()

arr3d[0]=42

arr3d

arr3d[0] = old_values

arr3d[0]

arr3d[1,0]

x = arr3d[1]

x

x[0]

arr

arr[1:6]

arr2d

arr2d[:2]

arr2d[:2,1:]

lower_dim_slice = arr2d[1,2:]

lower_dim_slice

lower_dim_slice.shape

arr2d[:,2:]

arr2d[:2,2:]

arr2d[:2,2]

arr2d[:2,1:]=0

arr2d

### 4.1.5 불리언 값으로 선택하기

names = np.array(['Bob','joe','Will','Bob','Will','Joe','Joe'])

data= np.array([[4,7],[0,2],[-5,6],[0,0],[1,2],[-12,-4],[3,4]])

names

names.shape

data

names =="Bob"

data [names=='Bob']

data [names =="Bob",1:]

data[names=="Bob",1]

names!= "Bob"

~(names == 'Bob')

data[~(names =="Bob")]

cond = names =="Bob"

data[~cond]

#세 가지 이름 중 두 가지 이름을 선택하려면 &(and) 와 |(or) 을 사용해주자

mask = (names =="Bob") | (names =="Will")

mask

data[mask]

data[data<0]=0

data

data[names != 'Joe'] =7

data

### 4.1.6 팬시 색인

arr = np.zeros((8,4))

arr

for i in range(8):
    arr[i]=i

arr

arr[[4,3,0,6]]

arr[[-3,-5,-7]]

arr = np.arange(32).reshape(8,-1)

arr

arr[[1,5,7,2],[0,3,1,2]]

arr[[1,5,7,2]][:,[0,3,1,2]]

arr[[1,6,7,2],[0,3,1,2]]

arr[[1,6,7,2],[0,3,1,2]]=0

arr

fancy_arr =arr[[1,6,7,2],[0,3,1,2]]

fancy_arr =12

arr

### 4.1.7 배열 전치와 축 바꾸기

# 배열 전치는 데이터를 복사하지 않고 데이터의 모양이 바뀐 뷰를 변환하는 특별한 기능
arr= np.arange(15).reshape((5,3))

arr

arr.T

np.dot(arr.T,arr)

arr.T

arr.T

#@ 연산자는 행렬 곱셈을 수행하는 또 다른 방법

arr.T @ arr

#.T는 축을 뒤바꾸는 특별한 경우

#ndarray = swapaxes 는 메서드를 통해 두 개의 축 번호를 받아 배열을 뒤바꿈

arr

arr.swapaxes(0,1)

arr

## 4.2 난수 생성

samples  = np.random.standard_normal(size=(4,4))
#표준정규분포로부터, 크기 4x4 표본 생성

samples

#random 모듈은 한 번에 하나으ㅢ 값만 생성

from random import normalvariate

N = 1_000_000

%timeit samples = [normalvariate(0, 1) for _ in range(N)]

%timeit np.random.standard_normal(N)

rng = np.random.default_rng(seed=12345)

data= rng.standard_normal((2,3))

type(rng)

## 4.3 유니버설 함수 : 배열의 각 원소를 빠르게 처리하는 함수

## ufunc 라고도 부름 -> 데이터 원소별로 연산을 수행

arr = np.arange(10)

arr

np.sqrt(arr) #각 원소의 제곱근을 계산 **0.5

np.exp(arr) #각 원소에서 지수 ex 를 계산한다.

x = rng.standard_normal(8)

y = rng.standard_normal(8)

x

y

np.maximum(x,y)

arr = rng.standard_normal(7)*5

arr

remainder, whole_part = np.modf(arr)

remainder

whole_part

arr

out = np.zeros_like(arr)

out

np.add(arr,1)

np.add(arr,1,out=out)

out

## 4.4 배열을 이용한 배열 기반 프로그래밍

points = np.arange(-5,5,0.01)

points

xs,xy = np.meshgrid(points,points)

xs

xy

x= np.linspace(1,10,10)

y = np.linspace(11,20,10)

x

y

X,Y = np.meshgrid(x,y)

import matplotlib.pyplot as plt

plt.scatter(X,Y)

plt.grid

z= np.sqrt(xs **2 + xy **2)

z

plt.imshow(z,cmap=plt.cm.gray,extent=[-5,5,-5,5])

plt.colorbar()

### 4.4.1 배열 연산으로 조건부 표현하기

xarr = np.array([[1.1,1.2,1.3,1.4,1.4]])

yarr= np.array([[2.1,2.2,2.3,2.4,2.4]])

cond = np.array([[True,False,True,True,False]])

result = [x if c else y for x, y, c in zip(xarr[0], yarr[0], cond[0])]

result

result = np.where(cond,xarr,yarr)

result

arr= rng.standard_normal((4,4))

arr

arr>0

np.where(arr>0,2,-2)

np.where(arr>0,2,arr)

### 4.4.2 수학 메서드와 통계 메서드

arr= rng.standard_normal((5,4))

arr

arr.mean()

np.mean(arr)

arr.sum()

np.sum(arr)

arr.mean(axis =1) #열

arr.sum(axis=0) #행

