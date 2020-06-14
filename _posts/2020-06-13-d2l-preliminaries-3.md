---
layout: post
title:  "D2L - Preliminaries - 3"
date:   2020-06-13T14:25:52-05:00
author: the-realest-one
categories: D2L
tags:	AI D2L
cover:  "/assets/North.jpg"
use_math: true
---

# D2L - Preliminaries

자 다시 한 번 recap 해보자. 우리의 목표는 좋은 모델을 만드는 것이다. 바꿔 말하면, loss function 의 값을 작게 하는 모델을 만드는 것이다.
궁극적으로는! unseen 데이터에도 성능이 좋은 일반적인 모델을 만드는 것이다!

그런데 당연하게 들리지만 사실 뭔가 어이 없는 말이다. 우리가 지금까지 본 것들로 학습을 하는데, 어떻게 안 본 것들도 잘하는가!
배운 것도 잘하기 힘든 세상인데 말이다. 프로그래머들은 하나 가르치면 열을 깨우치는 마법 지팡이를 원함이 틀림없다. 욕심쟁이들.

어쨌든, 그걸 하기 위해 model 을 fit 하는 것을 두 개의 중요 포인트로 나눈다고 한다.
i) 최적화: 모델을 observed data 에 fit 함
ii) 일반화: 수학적 원리와 사람의 지혜를 이용해, 우리가 train 에 쓴 데이터 이외의 데이터에도 성능이 잘 나오게 모델을 만드는 것!

이렇게 말을 하는데.... 이게 가능한가 싶다. 미적분이랑 무슨 관련이 있는지도 모르겠고. 하지만 이 말을 이 챕터의 가장 앞에 쓴 이유가 있을 것이다.
미적분과 관련을 알게 되면, 다시 돌아와서 이 챕터를 배우는 목적에 대해 쓰겠다.

자 다 읽고 돌아와서, 미적분이 왜 D2L 의 내용에 있는지 한 번 써보겠다.
일단, 이 미적분 챕터는 i) 최적화에 쓰이는 것 같다. 우리의 Loss function 은, 변수가 우리의 parameter 들이면서 (다변수함수), 합성함수 일 것이다.
사실 모든 함수는 합성함수로 나타낼 수 있으므로, 뭐 일반적인 복잡한 함수일 것이다.
각각 parameter 에 대해 loss function 을 미분하면, parameter 를 얼만큼 바꾸면 loss 가 얼만큼 줄지 대충 각이 나오고, 그 방향으로 parameter 를 바꿔주는 것이다.
우리는 데이터를 행렬, 벡터로 나타냈으므로 행렬에 대한 미분도 배우고, 다변수함수이므로 partial derivative 를 배우고, 그것들을 묶은 gradient 도 배운다!

즉 linear algebra 로, 우리의 모델의 prediction 과 실제 target 이 얼만큼 차이나는지 알고 또 이걸 작게할 기법을 생각한다면,
Calculus 가 우리 모델의 parameter 를 얼만큼 바꿔서 prediction 을 뱉게 할 것이냐. 뭐 이정도로 정리 가능할 것 같다.

## Calculus

#### Derivatives and Differentiation
고등학교 때 배운 미분 식을 다시 쓰지는 않겠다. 다만 앞으로의 표기를 위해 이건 써두자.
Given $y = f(x)$, 다음 식은 모두 equivalent 하다.

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

#### Partial Derivatives

위 단락에서 본 것은, 변수가 하나인 일변수 함수와 그의 미분이었다. 하지만, 실제 일반적인 함수들은 당연히 영향을 주는 변수가 많다!
이 친구들을 다변수함수라고 부른다.

$y = f(x_1, x_2, \ldots, x_n)$ 에 대해,
The *partial derivative* of $y$ with respect to its $i^\mathrm{th}$  parameter $x_i$ 는 다음과 같다.

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$

그리고 하나 더. 이것들 또한 모두 equivalent 하다

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = f_{x_i} = f_i = D_i f = D_{x_i} f.$$

#### Gradients

gradient vector 란, multivariate function 의 각 partial derivatives 들을 concat 한 벡터이다.
함수 $f: \mathbb{R}^n \rightarrow \mathbb{R}$ 에 대해 생각해보자.
이 함수의 input 은 $\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$ 이다. (row vector 의 T 를 했으므로 column vector 이다.)
이 함수의 output 은 scalar 값이다.
이 때 gradient of the function $f(\mathbf{x})$ with respect to $\mathbf{x}$ 는 다음과 같다.

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top,$$

정의는 이렇고, $n$-dimensional vector $\mathbf{x}$ 에 대해 다음과 같은 성질들이 소개된다.

1. For all $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$,
1. For all  $\mathbf{A} \in \mathbb{R}^{n \times m}$, $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$,
1. For all  $\mathbf{A} \in \mathbb{R}^{n \times n}$, $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$,
1. $\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$.

1,2,3 에 대한 증명은 다음 링크를 첨부한다. https://datascienceschool.net/view-notebook/8595892721714eb68be24727b5323778/
4 에 대한 증명은, 3 의 증명에 나오는 전개식을 보면서 생각할 수 있다.

Similarly, for any matrix $\mathbf{X}$, we have $\nabla_{\mathbf{X}} \|\mathbf{X} \|_F^2 = 2\mathbf{X}$.
이식은 미분을 하는 쪽이 행렬이다. 이것도 나중에 다시 찾아보자 // TODO

#### Chain Rule

Chain Rule 은, 우리가 미분하고자 하는 함수가 합성함수일 때, 미분을 쉽게 할 수 있도록 도와준다! 함수 $y=f(u)$ 과 $u=g(x)$ 가 둘 다 미분 가능할 때, then the chain rule states that

$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$

조금 더 일반적인 상황을 보자. 미분가능함수 $y$ 는 $u_1, u_2, \ldots, u_m$ 들에 대한 함수이다.
그런데 각 $u_i$ 들은 미분 가능하고, $u_i$ 들은 $x_1, x_2, \ldots, x_n$ 에 대한 함수이다.
이러면 $y$ 는 $x_1, x_2, \ldots, x_n$ 에 대한 함수가 될 수 있다.
또한, 체인 룰에 의해 도출되는 식은 다음과 같다.

$$\frac{dy}{dx_i} = \frac{dy}{du_1} \frac{du_1}{dx_i} + \frac{dy}{du_2} \frac{du_2}{dx_i} + \cdots + \frac{dy}{du_m} \frac{du_m}{dx_i}$$

for any $i = 1, 2, \ldots, n$.

## Automatic Differentiation

미분이 딥러닝 최적화의 핵심인 건 맞다. 하지만, 사람이 직접 모두를 할 필요도 없고, 실수도 많이 함.
그러니까, 파이토치 라이브러리에서 자동으로 해주는 미분을 쓰자!

다른 라이브러리는 오토미분 쓰려면, symbolic graph 를 만들어줘야 하는데, PyTorch 는 그냥 일반적인 imperative code 쓰면서 된다고 한다.
우리는 그냥 모델을 만들고, 데이터를 전달한다. 그러면 파이토치가 알아서 어떤 데이터가 어떤 연산을 통해 결합되어 output 을 뱉는지 그래프를 즉석에서 만든다고 한다!
이렇게 만들어진 그래프로, 파이토치가 gradient 를 backpropagate 할 수 있다.

여기에서는 backpropagate 를 그렇게 자세히 설명해주지는 않는다.
그냥 파이토치가 자동으로 만든 계산 그래프를 역추적하면서, 모든 파라미터마다 partial derivative 를 계산해 채워넣는 것. 이라고만 나온다.
backprop 을 나중에 따로 포스팅할지 안할지는 모르지만, 뭐 일단 넘어가자.

#### Simple Example

column vector $\mathbf{x}$ 에 대한 함수 $y = 2\mathbf{x}^{\top}\mathbf{x}$ 의 gradient 를 구해보자.

그 전에 Note 로 알려주는 게 있다. i) y 의 그래디언티를 계산하기 전에, 그걸 담아 놓을 공간을 미리 만들어놔야한다.
우리는 gradient 를 자주 계산하고 업데이트 할 것이다. Gradient 를 계산할 때마다 메모리를 새로 할당하면, 금방 oom 가 난다고 한다.
ii) vector $\mathbf{x}$ 에 대한 함수 $y$ 가 스칼라를 뱉는 함수여도, y 의 $\mathbf{x}$ 에 대한 그래디언트는 vector 이고
$\mathbf{x}$ 와 같은 shape 을 가진다.

그런데 코드를 보다보면, i) 가 이미 `requires_grad=True` 로 해결되는 것 같은데, 정확히는 모르겠다. // TODO

```python
x = torch.tensor([0.,1.,3.,5.], requires_grad=True)
y = 2 * torch.dot(x, x)

x.grad # 아무것도 안나옴.
y.backward() 
x.grad # tensor([ 0.,  4., 12., 20.])

x.grad == 4 * x # tensor([True, True, True, True]) 우리가 y 로 둔 것을 미분하면, 4x 가 맞다.
```

Note 하나 더. 파이토치는 기본적으로 grad 를 accumulate 하게 더한다고 한다.
그래서 만약 위에서 쓴 $mathbf{x}$ 를 변수로 하는 $z$ 라는 다른 함수의 gradient 를 구하고 싶으면,
예전 값인 x.grad 를 먼저 clear 해줘야 한다고 한다.

```python
x, x.grad # (tensor([0., 1., 3., 5.], requires_grad=True), tensor([ 0.,  4., 12., 20.]))

x.grad.zero_()
x.grad # tensor([0., 0., 0., 0.])

z = x.sum()
z # tensor(9., grad_fn=<SumBackward0>)

z.backward()
x.grad # tensor([1., 1., 1., 1.])
```

#### Detaching Computation

특정 함수 또는 변수를, computational graph 에서 빼고 계산하고 싶을 때 쓰는 기법이다.
이런 일이 있을지는 모르지만, 뭐 수학적 일반적으로 생각해야하므로 충분히 있을 수 있는 상황이다. 다음 상황을 보자.

$y$ 가 $x$ 의 함수라고 해보자. 그리고, $z$ 는 $y$ 와 $x$ 의 함수이다. 즉, $z$ 는 $y$ 의 함수이며서 $x$ 의 함수이다.
우리는 $z$ 의 $x$ 에 대한 그래디언트를 구하고 싶다. 그런데, $y$ 를 상수로 두고 싶다!
글로는 잘 이해가 안된다. 코드를 보자.

```python
x.grad.zero_()
y = x * x
y # tensor([ 0.,  1.,  9., 25.], grad_fn=<MulBackward0>)

u = y.detach() # !!! y 랑 같은 값을 가지는 tensor 를 뱉지만, y 가 computational graph 에서 어떻게 계산됐는지는 버린다!
#  
z = u * x
u, z # (tensor([ 0.,  1.,  9., 25.]), tensor([  0.,   1.,  27., 125.], grad_fn=<MulBackward0>))

z.sum().backward()
x.grad # tensor([ 0.,  1.,  9., 25.])
x.grad == u # tensor([True, True, True, True])
```

결과가 보이는가? `u = y.detach()` 를 함으로써, y 가 그래프에서 어떻게 계산됐는지는 빼고, 값만 그대로 복사한 u 가 만들어졌다.
이러면, the gradient will not flow backwards through u to x.
즉, `autograd.record` scope 밖에서 u 를 x 에 대한 함수로 계산한 것과 같은 효과이다. -> 모든 backward() call 에서 u 는 상수로 취급된다.
그래서 `z.sum().backward()` 에서, partial derivative 는 z = x * x * x 로 보는 식이 아니라 u 를 상수로 두고 x 만 변수로 두고 계산됐다.

y 는 계산이 기록이 되어 있다, 즉 계산 그래프에 남아있다.
그래서, the derivative of y = x * x with respect to x 를 얻기 위해 y.backward() 를 부를 수 있다.

```python
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x # tensor([True, True, True, True])

x.grad.zero_()
u.sum().backward() # <-- RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
x.grad == 2 * x
```

#### Computing the Gradient of Python Control Flow

이건 정말 놀랐다. 그냥 `y = ~~x` 이런 것으로 이루어진 식이 아니라, 여러 control flow 를 지나는 함수도 gradient 를 이렇게 쉽게 구할 수 있게 했다니.
조건문, 반복문, 함수 등 여러 control flow 를 지나는 함수에 대해서도 gradient 를 구할 수 있단 것이다.
정말 너무 대단하고 편하면서도, 이걸 만든 페이스북 개발자들에게 경의를 표한다....

```python
def f(a):
    b = a * 2
    while b.norm().item() < 1000:
        b = b * 2

    if b.sum().item() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(1,), requires_grad=True)
d = f(a)
d.backward()

a.grad # (tensor([4096.]),
d # tensor([1629.1516], grad_fn=<MulBackward0>),
a # tensor([0.3977], requires_grad=True))

a.grad == (d / a) # tensor([True])
```

위 f(a) 를 분석해보면, input a 에 대해 piecewise linear 하다.
즉, for any a 에 대해, f(a) = k * a 를 만족시키는 scalar k 가 존재한다.
그리고 k 는 a 에 의해 결정된다. 마지막 `a.grad == (d / a)` 로, `a.grad == f(a) / a == k` 임을 알 수 있다!

#### Exercise

여기 Exercise 들은 한 번 보자. TODO
볼 가치가 충분히 있는 듯. 아직 이쪽 이해 부족해
