---
layout: post
title:  "D2L - Preliminaries - 1"
date:   2020-06-07T14:25:52-05:00
author: the-realest-one
categories: D2L
tags:	AI D2L
cover:  "/assets/North.jpg"
use_math: true
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
그리고 실습 코드는 MxNet 용이 아닌 Pytorch 용을 사용한다.

## Data Manipulation

실습코드를 내가 여기에 모두 적는 건 의미가 없을 것이다. 중요한 것만 짚고 넘어가자

#### ndarray
ndarray: n-dimensional array.

Axis 가 1 인 ndarray 는 수학의 vector, Aixs 가 2 인 ndarray 는 행렬과 매칭된다고 한다.
그리고 Aixs 가 3 이상인 ndarray 는 tensor 라는 수학 개념과 매칭된다고 한다.

#### .shape

shape 가 꽤나 중요하고, ML 학습할 때 잘 맞춰줘야 함을 많이 들었다. 이거 못 맞춰줘서 계속 학습에 실패한다거나... 그래서 한 번 정리해보자!

torch.shape 은 각 차원의 element 수를 리턴한다.
```text
For 0D array, return a shape tupe with only 0 element    (i.e. ())    # scala 는 0D array 임을 나타내기 위함. (5.5) 이런 ndarray.
For 1D array, return a shape tuple with only 1 element   (i.e. (n,))  # (1,n)-row vector 과 (n,1)-column vector 를 한번에 이렇게 표
For 2D array, return a shape tuple with only 2 elements  (i.e. (n,m))
For 3D array, return a shape tuple with only 3 elements  (i.e. (n,m,k))
For 4D array, return a shape tuple with only 4 elements  (i.e. (n,m,k,j))

a = torch.ones(32*(1,))
a
# array([[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[ 1.]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]])
a.shape
# (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
a.ndim
# 32

출처: https://stackoverflow.com/questions/47564495/what-does-numpy-ndarray-shape-do
```

그리고 torch.size() 는 total element 개수를 리턴한다. 이것에 맞는, 즉 곱셈이 가능한 숫자로 torch.reshape() 도 가능하다.
가장 좋은 건 .reshape(4,-1) .reshape(-1,3,2) 이렇게도 가능하다는 것이다. x.size 에 의해, 나머지 차원의 elem 개수는 자동으로 나오기 때문이다.

#### Operations

자 이렇게 지금까지, 우리가 갖고 놀 object 를 정의하고 특성을 살펴보았다.
그러면 이제, 이 object 들을 가지고 할 연산을 생각해보자!
그리고 조금 더 실용적으로 생각을 해보면, 우리는 데이터를 표현할 방식으로 ndarray 를 선택한 것이다.
우리의 목적은 데이터를 갖고 여러 연산을 해보는 것이다. ndarray 로 데이터를 표현했으니, 데이터의 연산을 위해 ndarray 의 연산을 정의해야하는 것이다

먼저 unary scalar operation 은 $f: \mathbb{R} \rightarrow \mathbb{R}$, 즉 real number -> real number mapping 으로 나타낼 수 있다.
binary scalar operation 은 $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$ 로 표현이 가능할 것이다.

하지만 우리는 vector 의 연산이 궁금하다. 가장 쉬운 방법은, vector 끼리의 연산을 element-wise 연산으로 생각하는 것이다.
이것으로 하면, 우리는 scalar function 을 그대로 올려 $F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$ 를 정의할 수 있다
즉, $\mathbf{c} = F(\mathbf{u},\mathbf{v})$ 이런 함수를 $c_i \gets f(u_i, v_i)$ for all $i$, ($f$ 는 binary scala operation) 으로 정의하는 것이다.

당연히, 이런 벡터 연산을 모두 element-wise 로 정의할 필요는 없다. 이외에도 내적, 행렬곱 등의 연산들이 등장할 것이다.

#### Broadcasting

바로 위의 Operations 소챕터에서, element-wise vector binary operation 을 $F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$ 로 정의했다.
잘 살펴보면, input 2 개의 차원이 같은 것을 볼 수 있다.
이를 컴퓨터의 ndarray 로 끌고와 생각해보면, 우리가 할 ndarray 의 연산도 shape 가 같아야 함을 예상할 수 있다. 

하지만! 인간은 그렇게 모든 걸 다 고민하기엔 시간이 아깝다. 그러므로, broadcasting 이란 걸 도입했다!
Broadcasting 이란, input ndarray 들의 원소들을 그대로 확장해, 같은 shape 이 되게 만들어주고, element-wise 연산을 수행하는 것이다!

```python
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))

a + b
# tensor([[0, 1],
#         [1, 2],
#         [2, 3]])
```

#### Saving Memory

나도 완전히 몰랐던 게 있어서, 한 번 자세히 써보겠다! 여기서 x, y, z 는 모두 ndarray 이다.
```python
before = id(y)
y = y + x
id(y) == before
# False
```
같은 y 변수인데 왜 주소가 다른 결과가 나오는가? 파이썬은 y = y + x 를 두 번에 나누어 처리하기 때문이라고 한다.
먼저 y + x 를 계산하며 새로운 주소에 이 값을 넣고, 그리고 y 가 그 주소를 가리키게 하는 것이다!

```text
value 는 사실 ndarray 이다

Before
variable    Address      value
   x           0           3
   y           4           2

After
variable    Address      value
   x           0           3
               4           2
   y           8           5
```

하지만, 이건 2 개이 이유에 의해 안좋다고 한다.
1. memory allocation 을 매번 할 필요가 없음
2. 여러 변수가 하나의 parameter 를 가리키고 있었을 수도?

1 에 대하여. 기본적으로 memory allocation 은 꽤나 느린 연산이다.
그리고 ML 의 관점에서 보자면, 우리는 수백 메가바이트의 parameter 들을 가진 모델을 쓸 수도 있다.
심지어, 이 parameter 들을 매우 자주 업데이트해줘야 할 수도 있다! 어마어마하게 많은 메모리 손실과 시간 손실이 있을 것이 예상이 된다.

2 에 대하여. x, y, z 하나의 parameter 를 가리켰다고 해보자. 그런데 우리가 x 에 대해 연산을 하는데 결과에 새로운 주소를 할당했다면?
실제 parameter 는 업데이트됐지만, y, z 는 예전 것들을 가리키고 있을 것이다.
물론 이것이 원래 의도였을 수도 있다. 하지만 반대의 경우엔 슬퍼질 것이므로, 잘 구분해서 사용하자.

그래서, memory 를 그대로 이용하는 in-place 연산을 하려면 어떻게 해야하는가!
slice notation 을 쓰거나, 혹은 +=, -= 등의 연산을 하며 바로 대입하는 연산자를 쓰면 된다.

```python
z = torch.zeros_like(y)
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))
# Same id!

before = id(x)
x[:] = x + y
id(x) == before
# True
```

#### Conversion to Other Python Objects

torch.tensor 와 numpy.ndarray 간의 변환은 쉽다.
하지만 변환된 두 오브젝트가 메모리를 공유하지 않는다고 한다. 그래서 다음과 같은 불편이 조금 발생한다고 한다.
`when you perform operations on the CPU or on GPUs, you do not want MXNet to halt computation, waiting to see whether the NumPy package of Python might want to be doing something else with the same chunk of memory.`

그런데 무슨 이야기인지 아직 잘 모르겠어서, TODO 로 남긴다.

#### Exercise

Replace the two ndarrays that operate by element in the broadcasting mechanism with other shapes, e.g., three dimensional tensors. Is the result the same as expected?
라는 exercise 가 있어서 한 번 해봤는데, 전혀 모르겠다. 이것도 TODO

## Data Preprocessing

지금까지는, ndarray 에 예쁘게 담겨 있는 데이터를 갖고 노는 것을 연습했다.
하지만, real world 데이터는, 처음부터 그렇게 ndarray 에 예쁘게 담겨 있지 않다.
그래서, pandas 로 preprocessing 해서 ndarray 에 담는 것을 배워보자!
역시 이번 챕터에서도, 세세한 코드들을 다 쓰진 않는다. 정리할 것만 정리해보자.

#### Handling Missing Data

Pandas 로 불러왔을 때 NaN, None, null 등 비어 있는 데이터가 있다. 어떻게 할까?
여러 방법이 있지만, 책은 2 개의 방법을 소개한다.

*imputation*: missing value 를 채워넣는것.

*deletion*: missing value 를 무시한다.

pd.get_dummies 를 이용해, Categorical 컬럼을 value 에 따라 나누며 numerical column 으로 찢을 수 있다.
그리고 이렇게 데이터가 모두 numerical column 이 되면, ndarray 로 치환할 수 있다!


##### Exercise
Exercise 중에 `Delete the column with the most missing values` 가 있다. 재밌어 보이니 한 번 해보자.

```python
max_nan_count = -1
max_nan_col = ''

nan_counts = data.isna().sum() # pd.DataFrame 의 (column name, NaN 개수) 의 pd.Series 리턴.
for i, v in nan_counts.items():
    if v > max_nan_count:
      max_nan_col = i
      max_nan_count = v

df = data.drop(max_nan_col, axis=1) # NaN 이 가장 많은 컬럼 삭제
```
