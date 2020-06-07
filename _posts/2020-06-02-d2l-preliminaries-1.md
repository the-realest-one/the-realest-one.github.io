---
layout: post
title:  "D2L - Introduction"
date:   2020-05-26T14:25:52-05:00
author: the-realest-one
categories: D2L
tags:	AI D2L
cover:  "/assets/North.jpg"
---

# D2L - Preliminaries

이 챕터에서는 Data Manipulation, Data Preprocessing, Linear Algebra, Calculus, Automatic Differentiation, Probability, 그리고 도큐먼트 읽는 방법에 대해 서술한다.
챕터 이름이 Preliminaries 인 만큼, 머신러닝을 공부하기 위해서 이정도는 알아야지! 싶은 것만 모아뒀다고 한다.
가장 좋은 점은, 인트로에서 위의 것들이 머신러닝에 왜 필요하고 언제 필요한지, 당위성을 스토리 텔링으로 알려주는 점이다.

모든 머신러닝은 기본적으로 데이터에서 정보를 뽑아내는 것과 관련이 있다. 그래서, 우리는 코드로 데이터를 다루고, 변환하는 방법에 대해 알아야 한다.
선형대수를 통해, 우리는 tabular data 를 다루는 테크닉과 수학적 기반을 가질 수 있다.

그리고 또한, 딥러닝은 결국 최적화이다. 우리의 데이터에 가장 잘 fit 하는 parameter 를 찾는 게 우리의 목표이다.
파라미터를 움직이는 방향을 결정할 때 Calculus 가 필요하고, 이를 편하게 해주는 Automatic Differentiation 패키지를 훑는다.

그리고 머신러닝은 예측과 관련이 있다. 학습 데이터를 통해, 다른 데이터의 특정 attribute 의 값을 예측하는 것이니까 말이다.
불확실한 상황에서 그 값을 엄격하게 추론하려면, 확률을 알아야 한다.

이제부터는 코드 실습이 나오고, d2l 은 jupyter notebook 을 제공하고 있다. 나는 colab 을 이용해 실습해보겠다.

## Data Manipulation

실습코드를 내가 여기에 모두 적는 건 의미가 없을 것이다. 중요한 것만 짚고 넘어가자

### ndarray
ndarray: n-dimensional array.
MXNet 을 쓴다. -> ndarray with asynchronous computation on CPU,GPU, Distributed cloud architecture. And support auto differentiation.

Axis 가 1 인 ndarray 는 수학의 vector, Aixs 가 2 인 ndarray 는 행렬과 매칭된다고 한다.
그리고 Aixs 가 3 이상인 ndarray 는 tensor 라는 수학 개념과 매칭된다고 한다.

### .shape

shape 가 꽤나 중요하고, ML 학습할 때 잘 맞춰줘야 함을 많이 들었다. 이거 못 맞춰줘서 계속 학습에 실패한다거나... 그래서 한 번 정리해보자!

np.shape 은 각 차원의 element 수를 리턴한다.
```text
For 0D array, return a shape tupe with only 0 element    (i.e. ())    # scala 는 0D array 임을 나타내기 위함. (5.5) 이런 ndarray.
For 1D array, return a shape tuple with only 1 element   (i.e. (n,))  # (1,n)-row vector 과 (n,1)-column vector 를 한번에 이렇게 표
For 2D array, return a shape tuple with only 2 elements  (i.e. (n,m))
For 3D array, return a shape tuple with only 3 elements  (i.e. (n,m,k))
For 4D array, return a shape tuple with only 4 elements  (i.e. (n,m,k,j))

a = np.ones(32*(1,))
a
# array([[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[ 1.]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]])
a.shape
# (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
a.ndim
# 32

출처: https://stackoverflow.com/questions/47564495/what-does-numpy-ndarray-shape-do
```

그리고 np.size 는 total element 개수를 리턴한다. 이것에 맞는, 즉 곱셈이 가능한 숫자로 np.reshape() 도 가능하다.
가장 좋은 건 .reshape(4,-1) .reshape(-1,3,2) 이렇게도 가능하다는 것이다. x.size 에 의해, 나머지 차원의 elem 개수는 자동으로 나오기 때문이다.
## Data Preprocessing

## Linear Algebra