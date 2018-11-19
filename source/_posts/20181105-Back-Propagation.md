---
title: <Neural Network> 인공신경망에 대한 이해(Part 1 - Feedforward Propagation)
categories:
  - Danial Nam
tags:
  - Deep Learning
  - Neural Network
  - Artificial Intelligence
date: 2018-11-05 14:16:52
thumbnail:
mathjax: true
---

<br>

이번 포스트(Part 1)에서는 TensorFlow로 DNN을 구현하기 전에 먼저 기본 개념을 알아보고 다음 포스트(Part 2)에서는 인공 신경망을 가능하게 한 **Back Propagation** 에 대해 알아보도록 하겠다.(모바일에는 최적화되어있지 않으니 가능하면 PC로 보시길 추천한다)

<div style='display: none;'>
<img src="/images/danial/back-prop/thumbnail.png">
</div>

### 목차
- <a href='#nn-history'>Neural Network 역사 및 Back propagation의 중요성</a>
- <a href='#feedforward-propagation'>Feedforward Propagation 설명</a>
  - <a href='#init-network'>네트워크 초기화</a>
  - <a href='#layer1'>Layer 1 (input -> J)</a>
  - <a href='#layer2'>Layer 2 (J -> K)</a>
  - <a href='#layer3'>Layer 3 (K -> output)</a>
  <!-- - <a href='#conclusion'>정리</a> -->
- <a href='#cost-function'>오차 함수(Cost function)</a>
- <a href='#optimization'>가중치 최적화</a>


<h3 id='nn-history' href='#nn-history'>Neural Network 역사</h3>

 1943년, 워런 맥컬록(Warren McCulloch)와 월터 피츠(Walter Pitts)의 수학과 임계 논리(Threshold logic)라 불리는 알고리즘을 바탕으로 신경망을 위한 계산한 모델이 만들어지며 신경망 연구의 초석을 닦으며 **Neural Network** 역사가 시작되었다.

 하지만, 1969년에 마빈 민스키(Marvin Minsky)와 시모어 페퍼트(Seymour Papert)에 의해 기계학습 논문이 발표된 후 침체되었는데, 그 이유는 두 가지였다.

1. **단층 신경망** 은 선형으로 분류하기 때문에 아래의 그림처럼 문제가 배타적 논리합 회로(XOR problem)인 경우에는 해결하지 못한다.
<div>
<img style='width: 50%;' src='https://cdn-images-1.medium.com/max/2000/0*qdRb80zUpJPtrbRD.'/>
<a style='display: block; text-align: center;' href='https://medium.com/@jayeshbahire/the-xor-problem-in-neural-networks-50006411840b'>
출처 : [Medium] Back-Propagation is very simple. Who made it Complicated ?</a>
</div>

2. Computing power가 부족하다.

위의 두 가지 문제점 외에도 가중치를 업데이트하기 위한 회로 수학 계산이 너무 복잡하였기에 **오차역전파법(Back Propagation)** 이 세상에 나오기 전까지는 연구가 침체될 수밖에 없었다.

<h3 id='feedforward-propagation' href='#feedforward-propagation'>Feedforward Propagation 설명</h3>

<h4 style='color: #cb4a44;' id='init-network' href='#init-network'>네트워크 초기화</h4>
<img src="/images/danial/back-prop/network.png">

$$
Input = \left[ \begin{array}{cccc}
i_{1} & i_{2} \\\end{array} \right]
$$
$$
W_{ij} = \left[ \begin{array}{cccc}
W_{i1j1} & W_{i1j2} & W_{i1j3} \\
W_{i2j1} & W_{i2j2} & W_{i2j3} \\\end{array} \right]
W_{jk} = \left[ \begin{array}{cccc}
W_{j1k1} & W_{j1k2} & W_{j1k3} \\
W_{j2k1} & W_{j2k2} & W_{j2k3} \\
W_{j3k1} & W_{j3k2} & W_{j3k3} \\ \end{array} \right]
W_{ko} = \left[ \begin{array}{cccc}
W_{k1o1} & W_{k1o2} \\
W_{k2o1} & W_{k2o2} \\
W_{k3o1} & W_{k3o2} \\ \end{array} \right]
$$
$$
Output = \left[ \begin{array}{cccc}
o_{1} & o_{2} \\\end{array} \right]
$$

<p>
이 포스트에서는 간단한 계산을 위해 Hidden layer가 2층 구조인 모형으로 초기화를 하였지만, 실제로는 layer마다 Neuron 개수나 Hidden layer의 층수는 우리가 알아내야 할 하이퍼 파라미터이다.
</p>
<p>
<strong>인공 신경망</strong> 알고리즘에는 여느 예측 문제와 마찬가지로 Feature에 해당하는 Input이 존재하고, 들어온 데이터들이 Hidden layer의 Neuron이 가진 <strong>하이퍼 파라미터에 의해 모양이 바뀌는 비선형 기저 함수</strong>
</p>
<blockquote>
<strong>기저 함수</strong> 는 위에서 설명한 배타적 논리합 회로(XOR problem)를 해결하기 위한 방법으로, Input데이터 x대신 &phi;(x)를 사용하는 것을 의미한다.
여기서 "<strong>하이퍼 파라미터에 의해 모양이 바뀌는</strong>"라는 수식어가 붙은 이유는 Hidden layer로 이동할 때 계산되는 <strong>행렬(Matrix)</strong> 과 <strong>Bias</strong> 에 의해서 계속해서 모양을 바뀌는 것을 의미한다. 가장 기본적인 <strong>Activation 함수</strong> 인 Logistic Sigmoid를 예로 보자면,
$$
Sigmoid = 1/(1+\mathrm{e}^{-(w_{ji}x + b_{j})})
$$
형태의 기저함수를 가지게 되는 것.
</blockquote>
<p>
에 의해서 비선형 문제를 해결할 수 있는 형태로 정보의 차원을 변경하며 전파하는 과정을 거친다. 마지막에는 Output으로 분류인 경우에는 Class 개수만큼, 회귀분석인 경우에는 1개의 실수값을 내보내는 것으로 마무리 된다.
</p>
<p>
<strong>역방향 전파(Back Propagation)</strong> 을 이해하기 위해서는 먼저 인공신경망의 계산 과정인 <strong>순방향 전파(Feedforward propagation)</strong> 를 이해해야 하므로 살펴보도록 하자.
</p>

<hr>
<h4 id='layer1' href='#layer1' style='color: #cb4a44; '>Layer 1 (Input -> J)</h4>

<div style='float:left; width:40%; margin-right: 10px;'>
  <img src="/images/danial/back-prop/layer_j.png">
</div>
<div style=''>

$$
\definecolor{first}{RGB}{10, 199, 139}
\definecolor{second}{RGB}{74, 144, 226}
\left[ \begin{array}{cccc}
i_{1} & i_{2} \\\end{array} \right] \times
\left[ \begin{array}{cccc}
\textcolor{first}{W_{i1j1}} & \textcolor{first}{W_{i1j2}} & \textcolor{first}{W_{i1j3}} \\
\textcolor{second}{W_{i2j1}} &
\textcolor{second}{W_{i2j2}} &
\textcolor{second}{W_{i2j3}} \\\end{array} \right] +
\left[ \begin{array}{cccc}
b_{j1} & b_{j2} & b_{j3} \\\end{array} \right]

$$
$$
\downarrow
$$
$$
\definecolor{first}{RGB}{10, 199, 139}
\definecolor{second}{RGB}{74, 144, 226}
\left[ \begin{array}{cccc}
\textcolor{first}{W_{i1j1}} \times i_{1} +
\textcolor{second}{W_{i2j1}} \times i_{2} + b_{j1} \\
\textcolor{first}{W_{i1j2}} \times i_{1} +
\textcolor{second}{W_{i2j2}} \times i_{2} +
b_{j2} \\
\textcolor{first}{W_{i1j3}} \times i_{1} +
\textcolor{second}{W_{i2j3}} \times i_{2} +
b_{j3} \\
\end{array} \right]^{T} =
\left[ \begin{array}{cccc}
J_{in1} \\ J_{in2} \\ J_{in3} \\\end{array} \right]^{T}
$$

<p style=''>
  계산은 어려운게 없으니 따로 설명하지 않겠다. 중요한 점은 Hidden layer는 들어온 값(<strong>J<sub>in</sub></strong>)과 나가는 값(<strong>J<sub>out</sub></strong>)이 다르다는 것이다. 그 과정은 아래에서 살펴보자.
</p>
</div>

<div class='clearfix' style='clear:both'>
<div style='float:left; width:40%; margin-right:10px;'>
  <img src="/images/danial/back-prop/activation_j.png">
</div>
<div style=''>

<p style=''>
<strong>Hidden layer</strong>의 <strong>Neuron</strong>은 앞서 설명했듯 <strong>기저함수</strong>의 역할을 해야하므로 좌측의 그림처럼 <strong>J<sub>in</sub></strong>이 Activation function에 의해 변한 <strong>J<sub>out</sub></strong>을 다음 레이어에 들어가는 Input이 되게 한다.
이 포스팅의 예제에서 <strong>활성화 함수(Activation function)</strong> 은 모두 <strong>Logistic Sigmoid</strong> 함수를 사용하기로 한다.
</p>

$$
Sigmoid = 1/(1+\mathrm{e}^{-x})
$$

$$
Sigmoid(J_{in}) =
\left[ \begin{array}{cccc}
1/(1+\mathrm{e}^{-J_{in1}}) \\
1/(1+\mathrm{e}^{-J_{in2}}) \\
1/(1+\mathrm{e}^{-J_{in3}}) \\\end{array} \right]^{T}
= \left[ \begin{array}{cccc}
J_{out1} \\ J_{out2} \\ J_{out3} \\\end{array} \right]^{T}
$$

<p style=''>
이제는 위에서 설명한 <strong>하이퍼 파라미터에 의해 모양이 바뀌는</strong>이라는 표현이 이해가 더 잘 될 것이다. <strong>J<sub>in</sub></strong>은 w<sub>ij</sub>와 b<sub>j</sub>에 의해 바뀌고 그에 의해 <strong>J<sub>out</sub></strong>이 바뀔 것이므로.
</p>
</div>
</div>

<hr>

<div class='clearfix'>
<h4 id='layer2' href='#layer2' style='color: #cb4a44; clear:both; margin-top: 10px;'>Layer 2 (J -> K)</h4>
<div style='float:left; width:40%; margin-right: 10px;'>
  <img src="/images/danial/back-prop/j_k_layer.png">
</div>
<div style=''>

$$
\definecolor{first}{RGB}{10, 233, 134}
\definecolor{second}{RGB}{74, 144, 226}
\definecolor{third}{RGB}{245, 166, 35}
\left[ \begin{array}{cccc}
J_{out1} & J_{out2} & J_{out3}\\\end{array} \right] \times
\left[ \begin{array}{cccc}
\textcolor{first}{W_{j1k1}} & \textcolor{first}{W_{j1k2}} & \textcolor{first}{W_{j1k3}} \\
\textcolor{second}{W_{j2k1}} & \textcolor{second}{W_{j2k2}} & \textcolor{second}{W_{j2k3}} \\
\textcolor{third}{W_{j3k1}} & \textcolor{third}{W_{j3k2}} & \textcolor{third}{W_{j3k3}} \\ \end{array} \right] +
\left[ \begin{array}{cccc}
b_{k1} & b_{k2} & b_{k3} \\\end{array} \right]
$$
$$
\downarrow
$$
$$
\definecolor{first}{RGB}{10, 233, 134}
\definecolor{second}{RGB}{74, 144, 226}
\definecolor{third}{RGB}{245, 166, 35}
\left[ \begin{array}{cccc}
\textcolor{first}{W_{j1k1}} \times J_{out1} +
\textcolor{second}{W_{j2k1}} \times J_{out2} +
\textcolor{third}{W_{j3k1}} \times J_{out3} + b_{k1}\\
\textcolor{first}{W_{j1k2}} \times J_{out1} +
\textcolor{second}{W_{j2k2}} \times J_{out2} +
\textcolor{third}{W_{j3k2}} \times J_{out3} + b_{k2}\\
\textcolor{first}{W_{j1k3}} \times J_{out1} +
\textcolor{second}{W_{j2k3}} \times J_{out2} +
\textcolor{third}{W_{j3k3}} \times J_{out3} + b_{k3}\\
\end{array} \right]^{T} =
 \left[ \begin{array}{cccc}
K_{in1} \\ K_{in2} \\ K_{in3} \\\end{array} \right]^{T}
$$

<p style=''>
  <strong>K<sub>in</sub></strong>과 <strong>K<sub>out</sub></strong> 사이의 Activation function은 Layer 1에서와 마찬가지로 Logistic Sigmoid를 사용한다.
</p>

$$
Sigmoid(K_{in}) =
\left[ \begin{array}{cccc}
1/(1+\mathrm{e}^{-K_{in1}}) \\
1/(1+\mathrm{e}^{-K_{in2}}) \\
1/(1+\mathrm{e}^{-K_{in3}}) \\\end{array} \right]^{T}
= \left[ \begin{array}{cccc}
K_{out1} \\ K_{out2} \\ K_{out3} \\\end{array} \right]^{T}
$$

<p style=''>
  Layer 1과 비교해서 새로운 점이 없으므로 그림상에 Notation은 자세하게 하지 않았다.
</p>
</div>
</div>

<hr>

<h4 id='layer3' href='#layer3' style='color: #cb4a44; clear:both; margin-top: 10px;'>Layer 3 (K -> output)</h4>
<div style='float:left; width:40%; margin-right: 10px;'>
  <img src="/images/danial/back-prop/k_o_layer.png">
</div>
<div style=''>

$$
\definecolor{first}{RGB}{245, 166, 35}
\definecolor{second}{RGB}{74, 144, 226}
\definecolor{third}{RGB}{189, 16, 224}
\left[ \begin{array}{cccc}
K_{out1} & K_{out2} & K_{out3}\\\end{array} \right] \times
\left[ \begin{array}{cccc}
\textcolor{first}{W_{k1o1}} & \textcolor{first}{W_{k1o2}} \\
\textcolor{second}{W_{k2o1}} & \textcolor{second}{W_{k2o2}} \\
\textcolor{third}{W_{k3o1}} & \textcolor{third}{W_{k3o2}} \\ \end{array} \right] +
\left[ \begin{array}{cccc}
b_{o1} & b_{o2} \\\end{array} \right]
$$

$$
\downarrow
$$

$$
\definecolor{first}{RGB}{245, 166, 35}
\definecolor{second}{RGB}{74, 144, 226}
\definecolor{third}{RGB}{189, 16, 224}
\left[ \begin{array}{cccc}
\textcolor{first}{W_{k1o1}} \times K_{out1} +
\textcolor{second}{W_{k2o1}} \times K_{out2} +
\textcolor{third}{W_{k3o1}} \times K_{out3} + b_{o1}\\
\textcolor{first}{W_{k1o2}} \times K_{out1} +
\textcolor{second}{W_{k2o2}} \times K_{out2} +
\textcolor{third}{W_{k3o2}} \times K_{out3} + b_{o2}\\
\end{array} \right]^{T} =
\left[ \begin{array}{cccc}
o_{in1} \\ o_{in2}\\\end{array} \right]^{T}
$$

<p style=''>
  Output으로 나가는 값(o<sub>out</sub>)은 우리가 예측하고자하는 Target 값을 가장 잘 보여줄 수 있는 형태로 만들어야한다.
</p>
<p style=''>
  일반적으로는 <strong>회귀분석</strong>에는 특별한 <strong>활성화 함수 없이</strong> 내보내는 경우가 있지만, 만약 Target 데이터가 0보다 크고 1보다 작은 실수값만을 가진 상황이라면 Logistic Sigmoid을 사용했을 때 더 좋은 결과가 나올수도 있다는 의미다.
</p>
<p style=''>
  이 포스트에서는 분류 문제라고 가정하여 활성화 함수로는 <strong>Softmax</strong>를 이용하여 확률값처럼 변환시키는 것을 예로 들겠다.
</p>


</div>
<div class='clearfix' style='clear:both'>
<div style='float:left; width:40%; margin-right: 10px;'>
  <img src="/images/danial/back-prop/softmax_o.png">
</div>
<div style=''>

$$
Softmax = \mathrm{e}^{o_{ina}}/(\sum_{a=1}^{2}\mathrm{e}^{o_{ina}})
$$

$$
Softmax(o_{in}) =
\left[ \begin{array}{cccc}
\mathrm{e}^{o_{in1}}/(\sum_{a=1}^{2}\mathrm{e}^{o_{ina}})   \\ \mathrm{e}^{o_{in2}}/(\sum_{a=1}^{2}\mathrm{e}^{o_{ina}})   \\\end{array} \right]^{T} = \left[ \begin{array}{cccc}
o_{out1} \\ o_{out2}\\\end{array} \right]^{T}
$$

<p style=''>
이렇게 Output으로 나온 o<sub>out</sub> 벡터가 우리의 예측값 y_pred이다.

물론 Random하게 초기화된 W, b값에 의해 예측한 값이라고 하기엔 터무니없는 값들이 나올 것임을 명심하자.
</p>

</div>
</div>

<hr>

<!-- <h3 id='conclusion' href='#conclusion' style='color: #cb4a44; clear:both; margin-top: 10px;'>정리</h3>
<p>
input 데이터와 output은 실재하는 측정된 값이고, 그 사이의 관계를 우리가 알아내는 것이 Machine learning의 목표이므로 우리가 "학습"시키는 대상은 바로 각 layer간에 망처럼 연결된 W, b이다.
</p>
<hr> -->

<h3 id='cost-function' href='#cost-function' style='clear:both; margin-top: 10px;'>오차 함수(Cost function)</h3>

로지스틱 활성 함수를 이용한 분류 문제를 풀 때는 정답 y가 클래스 k에 속하는 데이터에 대해서 k번째 값만 1이고 나머지는 0인 one-hot-encoding 벡터를 사용한다.

Cost-function은 Cross-Entropy Error를 사용한다.

$$
cross entropy = - (1/n)(\sum_{i=1}^{3} (y_{i} \times \log(o_{outi})) + ((1-y_{i}) \times \log((1-o_{outi}))))
$$

<h3 id='optimization' href='#optimization' style='clear:both; margin-top: 10px;'>가중치 최적화</h3>

오차함수를 최소화하기 위해 아래처럼 미분(gradient)을 사용한 <strong>Steepest gradient descent</strong> 방법을 적용하자. 여기서 &mu;는 step size 혹은 learning rate라고 부른다.

$$
w_{k+1} = w_{k} - \mu \frac{\partial C}{\partial w}
$$
$$
b_{k+1} = b_{k} - \mu \frac{\partial C}{\partial b}
$$

문제점은 단순하게 수치적으로 미분을 계산하게되면 모든 가중치에 대해서 개별적으로 미분을 계산해야하는 문제가 있다. 하지만 **역전파 (Back propagation)** 을 사용하면 모든 가중치에 대한 미분값을 한번에 계산할 수 있다.

다음 포스트에서 자세하게 알아보도록 하자!

> 이번 **<Neural Network> 인공신경망에 대한 이해** 포스팅들은 시간을 많이 들여서 가능한 쉽고 직관적으로 설명하기 위해 노력하였다. 만약 도움이 되었다면! 공유를 부탁드린다!

<h3 style='clear:both; margin-top: 20px;'> Related Posts</h3>

[Backpropagation calculus | Deep learning, chapter 4 by 3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8&fbclid=IwAR2lsWOByt_MrzBkv5-Dc9P6JIdvHv1pUELE5q-0SVqQ73b6tS-RYGUI9eM)
[신경망 기초 이론](https://datascienceschool.net/view-notebook/0178802a219c4e6bb9b820b49bf57f91/)
