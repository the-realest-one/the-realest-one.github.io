---
layout: post
title:  "D2L - Preliminaries - 2"
date:   2020-06-09T14:25:52-05:00
author: the-realest-one
categories: D2L
tags:	AI D2L
cover:  "/assets/North.jpg"
use_math: true
---

# D2L - Preliminaries

Preliminaries 가 너무 길어서, 여러 개의 글로 나눠서 쓸 예정이다.
이번 글에서는, 핵심 아이디어가 되는 Linear Algebra 를 공부해보자.

왜 ML 하는데 Linear Algebra 를 배우는가? 를 말해보겠다. 그리고 이건, D2L 의 linear algebra 파트를 다 읽고, 그 후에 내 글의 앞부분에 쓰는 것이다.
원래 뭐든지, 뭔가를 하는 이유가 타당하고 얻는 게 있다는 걸 느껴야 열심히 하지 않겠는가.

우리는 전 챕터로, 데이터를 tabular 하게 만들었다. 그러면 이제, 몇 가지의 트릭을 더 쓰면 데이터를 행렬 혹은 벡터로 바라볼 수 있게 될 것이다.
그러면 이제 무엇이 가능해지느냐! 우리가 원하는 것은 모델의 prediction 이 실제 target 에 가까워지는 것이다. 
target 과 prediction 을 벡터로 보면, 둘 사이의 거리를 정의할 수 있다(by *norm*).
그리고 거리를 줄이려는 여러 선형대수 기법들을 쓸 수 있다!!

## Linear Algebra

이번 챕터는 상당히 길다... 하지만 linear algebra 는 재밌고 위대한 학문이니까 한 번 공부해보자!
또한 머신러닝 입장에서 말하면, 논리의 근본이 되는 학문이기에 중요하다고 한다.
데이터를 ndarray 로 표현하려고 한 게, 선형대수의 논리를 쓰기 위해서가 아닐끼?
사실 난 아직, 선형대수가 ML 에 쓰이는 거는 행렬곱, PCA 등 밖에 모른다. 나중에는 선형대수가 진짜 ML 의 근본이 되는지 알고 싶다는 마음을 가지며, 시작해보자.

실수 스칼라 x 는 $x \in \mathbb{R}$ 로 나타낸다.
벡터의 정의는 `List of numbers` 이다. 벡터는 *point in a space* 로 생각할 수 있다. 혹은 크기와 방향, 혹은 공간의 점을 가리키는 화살표로 생각할 수 있다.
그리고 list of number 인 벡터 $\mathbf{x}$ 는 다음과 같이 쓸 수 있다.

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\x_{2}  \\ \vdots  \\x_{n}\end{bmatrix},$$

그리고 벡터 $\mathbf{x}$ 가 $n$ 실수 스칼라 값으로 이루어져 있으면, $\mathbf{x} \in \mathbb{R}^n$ 로 표현할 수 있다.

그리고 x.shape 은 위 챕터에서 말한대로, 각 axis 의 dimensionality 를 나타낸다.

#### Matrices

벡터는 스칼라를 0 차원에서 1 차원으로 끌어올리며 일반화했다.
행렬은 벡터를 1 차원에서 2차원으로 끌어올리며 일반화한 것이다.

텐서는 행렬을 표현하는 일반적인 방법이다.

elementwise 행렬곱은 *Hadamard product* 로 불린다고 한다. 수식으로는 다음과 같이 나타낼 수 있다,

$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

python code 에서는
```python
A = torch.arange(24).reshape(2, 3, 4)
B = torch.arange(24).reshape(2, 3, 4)
A * B 
```
로 구현되어 있다.

그리고, 예시로 이 코드를 한 번 보자.

```python
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
"""
(tensor([[[ 2,  3,  4,  5],
          [ 6,  7,  8,  9],
          [10, 11, 12, 13]],
 
         [[14, 15, 16, 17],
          [18, 19, 20, 21],
          [22, 23, 24, 25]]]),
 torch.Size([2, 3, 4]))
"""
```
결과를 보면, broadcasting 이 적용되고, tensor 끼리 binary elementwise operation 은 shape 를 바꾸지 않는 것을 알 수 있다.


#### Reduction

차원을 줄이는 것에 대해 집중해서 봐보자. 여기에서는 sum() 함수로 reduction 의 예시를 든다,
sum() 함수는, tensor 의 모든 축을 따라, scalar 값으로 reduce 한다. reduce 할 축을 정해줄 수도 있다.
axis 0, row dimension 으로 정해주면, axis 0 의 dimension 이 사라지며 output 의 shape 이 결정된다.

```python
A = torch.arange(20).reshape(5, 4)
"""
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19]])"""

A_sum_axis0 = A.sum(axis=0) # tensor([40., 45., 50., 55.])
A_sum_axis1 = A.sum(axis=1) # tensor([ 6., 22., 38., 54., 70.])
A.sum() # tensor(190.)
A.sum(axis=[0,1]) # tensor(190.)
```

mean() 도 똑같이 axis 를 정해주며 reduction 이 가능하다.

물론 non-reduction sum 도 가능하다. axes 를 안 바꾸고 하는 연산도, 필요할 때가 있을 것이다.
그리고 axis 를 따라 cumulative sum 도 가능하다. 이것 또한 cumulative sum 이므로, dimension 은 바뀌지 않는다.

```python
sum_A = A.sum(axis=1, keepdims=True)
"""
tensor([[ 6.],
        [22.],
        [38.],
        [54.],
        [70.]])"""

A.cumsum(axis=0)
"""
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  6.,  8., 10.],
        [12., 15., 18., 21.],
        [24., 28., 32., 36.],
        [40., 45., 50., 55.]])"""
```

#### Dot Product

드디어 내적이다!
기본적으로 내적은 두 벡터 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ 를 받아 scala $c \in \mathbb{R}$ 를 뱉는 연산이다.
그리고 이 c 는, 두 벡터간 같은 위치의 element 끼리 곱한 것들을 모두 더해 계산된다.
notation 은 $\mathbf{x}^\top \mathbf{y}$ (or $\langle \mathbf{x}, \mathbf{y}  \rangle$) 이러하다.

파이토치에서는 `torch.dot(x,y)` 로 구현되어 있다.
물론, 내적의 정의에 따라 elementwise multiplication -> sum 으로, `torch.sum(x * y)` 으로도 표현이 가능하다.

우리가 내적을 이렇게 정의했으므로, x, w 를 내적해 weighted sum 을 $\mathbf{x}^\top \mathbf{w}$ 으로 쉽게 생각할 수 있다.
만약 w.sum() == 1 인 w 였다면, 내적 $\mathbf{x}^\top \mathbf{w}$ 은 weighted average 가 될 것이다.
또한 고등학교 때 배운 내적식과 각도 식을 떠올려보자. unit vector 두 개를 내적하면, 내적 값은 사잇각 의 cosine 값이 된다.

#### Matrix-Vector Products

자 이제, 선형대수의 매우 아름다운 논리가 하나 나온다. 잘 지켜보자!
먼저 우리의 재료를 보자. $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{x} \in \mathbb{R}^n$ 이다.

똑똑한 논리 1. 행렬의 row vector 를 하나로 보며, 컬럼을 하나만 가진 것처럼 봐보자.

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

where each $\mathbf{a}^\top_{i} \in \mathbb{R}^n$ is a row vector representing the $i^\mathrm{th}$ row of the matrix $\mathbf{A}$.

이러면, 각 $\mathbf{a}^\top_{i} \in \mathbb{R}^n$ 이므로, x 와 같은 shape 을 가진다. 즉, 내적이 가능하다!
그러면 이제 broadcasting 을 적용해보자.

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

이렇게 표현할 수 있을 것이다. 그런데 내적은 스칼라를 뱉으므로, Ax 는 $\mathbb{R}^m$ 에 속하는 벡터가 되었다!
여기에서, 우리는 $\mathbb{R}^n$ 에 속하는 x 에 행렬 A 를 곱함으로써, $\mathbb{R}^m$ 으로 project 했다는 걸 관찰할 수 있다.
즉, 벡터에 행렬을 product 함으로써, shape 을 바꿀 수 있는 것이다!

```python
A.shape, x.shape, torch.mv(A, x)
# (torch.Size([5, 4]), torch.Size([4]), tensor([ 14.,  38.,  62.,  86., 110.]))
```

#### Matrix-Matrix Multiplication

자 위에서 했던 것처럼, 행렬을 row 를 하나만 가진 것처럼, 그리고 column 을 하나만 가진 것처럼 보고, broadcasting 까지 하면 행렬곱도 쉽게 이해할 수 있다.

먼저 우리의 재료를 보자. $\mathbf{A} \in \mathbb{R}^{n \times k}$ and $\mathbf{B} \in \mathbb{R}^{k \times m}$:

여기에서 A 행렬의 row vector 를 하나의 원소를 보며 컬럼을 하나만 가진 것처럼 보자.
또 B 행렬의 column vector 를 하나의 원소로 보며, row 를 하나만 가진 것처럼 보자.
$\mathbf{a}^\top_{i} \in \mathbb{R}^k$, $\mathbf{b}_{j} \in \mathbb{R}^k$

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$

이렇게 표현하는 것이다.

그러면 the matrix product $\mathbf{C} \in \mathbb{R}^{n \times m}$ 는
각 원소 $c_{ij}$ 를 내적 $\mathbf{a}^\top_i \mathbf{b}_j$ 으로 계산하며 형성된다!

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

세 번째 식에서 네 번째 식으로 갈 때는 broadcasting 논리가 적용된다. 이렇게, 행렬곱이 완성된다!
나는 broadcasting 이, 단지 ML framework 들에서 만들어진 것이라고 생각했다.
그런제 지금 챕터를 보니, ML framework 에서 선형 대수의 논리를 구현하다보니, broadcasting 이 나올 수 밖에 없었을 거 같다는 생각이 든다.

```python
B = torch.ones(4, 3)
torch.mm(A, B)
"""
tensor([[ 6.,  6.,  6.],
        [22., 22., 22.],
        [38., 38., 38.],
        [54., 54., 54.],
        [70., 70., 70.]])"""
```

#### Norms

드디어 norm 이다! 선형대수학에서 norm 의 정의를 먼저 살펴보자. (선형대수학 이란 걸 괜히 강조한 것이 아니다. 다른 분야에서 norm 은 다르게 정의될 수 있다.)

$$\mathbf{V} is vector space on \mathbb{F} \\
f: \mathbf{V} \to \mathbb{F}. for \mathbf{u}, \mathbf{v} \in \mathbf{V} and \mathit{k} \in \mathbb{F} \\
1. f(\mathit{k} \mathbf{u}) = |\mathit{k}| f(\mathbf{u}). \\
2. f(\mathbf{u} + \mathbf{v}) \leq f(\mathbf{u}) + f(\mathbf{v}). \\
3. f(\mathbf{u}) \geq 0 and \mathbf{u} \eq 0 \iff f(\mathbf{u}) \eq 0 \\
$$

이렇듯, 기본적으로 norm 은 vector 를 받아 스칼라를 뱉는 함수다. 그리고, 정의에 따라 벡터들의 측정 혹은 비교를 가능하게 만들어준다.
선형대수나 해석학같은 수학을 처음 접하면, norm 이라고 배운 특정 함수들만 norm 이라고 생각하기 쉽다. Euclidean norm 혹은 맨해튼 norm 등 말이다.
하지만, 위 정의에만 맞으면 얼마든 norm 을 우리가 무수히 새롭게 정의할 수 있음을 인지해두자.

흔히 쓰이는 Euclidean norm, Manhattan norm 등은 $\ell_p$ *norm* 에 속한다. $\ell_p$ norm 의 계산은 다음과 같다.

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

벡터의 $\ell_2$ norms 과 비슷하게, $\mathbf{X} \in \mathbb{R}^{m \times n}$ 에게는 *Frobenius norm* 이 있다.
정의는 다음과 같다.

$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$

뭐 그 다음부터는, 선형대수는 ML 에 많이 쓰이고, 또 선형대수 with ML 로도 방대한 학문 분야가 있다 이런 말들을 한다.
일단 그렇구나 하고 넘어가자. 언젠가 내 블로그에는 `선형대수와 군` 공부 정리 글을 포스팅할 거니까 말이다.

#### Exercise

Exercise 가 재밌으니, 한 번 보자.

1. $(\mathbf{A}^\top)^\top = \mathbf{A}$ 증명. elementwise 로 하나씩 보면 자명하다.
1. $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$. 이것도 elementwise 로 직접 노트에 써보자. 자명하다.
1. Given any square matrix $\mathbf{A}$, is $\mathbf{A} + \mathbf{A}^\top$ always symmetric? 1, 2 에 의해 참이다.

len() 함수에 대하여.

```python
A = torch.arange(24, dtype = torch.float32).reshape(4, 3, 2)
len(A) # 4
len(A[0]) # 3
len(A[0][0]) # 2
```

len() 함수가 결과가 꽤나 신기하다. 약간 C 나 C++ 의 다중 array 처럼 tensor 를 다루나보다.
A.sum(axis = 0), A.sum(axis = 1), A.sum(axis = 2) 를 해보는 것도 재밌다. 정말 C++ 논리와 비슷하게 indexing 이 되는 듯 하다.
