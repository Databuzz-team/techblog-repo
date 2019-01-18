---
title: <Neural Network> 인공신경망에 대한 이해(Part 2 - Back Propagation)
categories:
  - Danial Nam
tags:
  - Deep Learning
  - Neural Network
  - Artificial Intelligence
date: 2018-12-27 15:54:51
thumbnail:
---

<div style='display: none;'>
<img src="/images/danial/back-prop/thumbnail_2.png">
</div>

<br>

이번 포스트(Part 2)에서는 인공 신경망을 가능하게 한 **Back Propagation** 에 대해 알아보도록 하겠다.(모바일에는 최적화되어있지 않으니 가능하면 PC로 보시길 추천한다)

만약 인공 신경망의 기본 개념과 **Feedforward Propagation** 에 대해서 잘 모른다면 [이전 포스트(Part 1)](https://Databuzz-team.github.io/2018/11/05/Back-Propagation/)를 먼저 보고 오기 바란다.

### 목차
- <a href='#why-back-propagation'>왜 Back propagation를 이해해야 할까?</a>
- <a href='#back-propagation'>Back Propagation 설명</a>
  - <a href='#derivatives'>역전파 계산에 사용될 도함수</a>
  - <a href='#layer1'>Back propagating the error (output Layer -> K Layer2)</a>
  - <a href='#layer2'>Back propagating the error (K Layer2 - > J Layer1)</a>
  - <a href='#v-g-p'>그래디언트 소멸 문제(Vanishing gradient problem)</a>
  - <a href='#initial-weight'>Weight 초기값 설정</a>
  - <a href='#conclusion'>문제점 해결 방법 및 결론</a>

<h3 id='why-back-propagation' href='#why-back-propagation'>왜 Back propagation를 이해해야 할까?</h3>

"어차피 TensorFlow를 사용하면 다 자동으로 계산해주는 것인데 왜 우리가 공부해야 하는 것일까?"

합리적인 질문이다. 공식만 봐도 어려워보이는 이 부분을 공부하는 것이 동기부여가 쉽게 되지않는것이 사실이기 때문에..

하지만, [Yes you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b) 글에서 설명했듯 이 **Back Propagation** 은 **Leaky Abstraction** 라는 것이다.

> **Leaky Abstraction**<br>
Joel Spolsky이 설명한 [The Law of Leaky Abstraction](https://www.joelonsoftware.com/2002/11/11/the-law-of-leaky-abstractions/)에서 사용된 표현으로써 프로그래밍 언어를 추상화시켜서 내부 구현을 모르도록 만들어놨지만, <strong style='color:red'>결국 제대로 사용하려면 내부 구현을 상당 부분 알아야 한다는 것을 의미한다.</strong>

이 포스트에서는 인공 신경망의 문제점에 대해서는 다루지 않기 때문에 왜 Leaky Abstraction이라 설명했는지 궁금한 경우에는 위의 [블로그 링크](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)를 참고하길 바란다.

<h3 id='back-propagation' href='#back-propagation'>Back propagation 설명</h3>
시작에 앞서 먼저 <a href='/2018/11/05/Back-Propagation/'>이전 포스트</a>에서 정의한 네트워크를 다시 살펴보자.

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

(자세한 Feedforward 계산은 <a href='/2018/11/05/Back-Propagation/'>이전 포스트</a>에서 확인하길 바란다.)
<br><br>
<blockquote>
이번 포스트는 계산식이 많이 나오고 첨자가 헷갈릴 수 있으니 공책에 적으며 따라가는 것을 추천한다.
</blockquote>


<h4 id='derivatives' href='#derivatives'>역전파 계산에 사용될 도함수</h4>
먼저 설명을 시작하기에 앞서 우리가 사용하게 될 도함수들을 살펴보자.

<strong>Sigmoid</strong>

$$
Sigmoid = 1/(1+\mathrm{e}^{-x})
$$

$$
\frac{\partial (1/(1+\mathrm{e}^{-x}))}{\partial x} = 1/(1+\mathrm{e}^{-x}) \times (1- 1/(1+\mathrm{e}^{-x}))
$$

$$
\frac{\partial Sigmoid}{\partial x} = Sigmoid \times (1- Sigmoid)
$$

<strong>Softmax</strong>

$$
Softmax = \mathrm{e}^{x_{a}}/(\sum_{a=1}^{2}\mathrm{e}^{x_{a}})
$$
$$
\frac{\partial (Softmax)}{\partial x_{1}} = \mathrm{e}^{x_{1}} * \mathrm{e}^{x_{2}}/ (\mathrm{e}^{x_{1}}+\mathrm{e}^{x_{2}})^2
$$

<strong>Cross Entropy</strong>

$$
cross entropy = - (1/n)(\sum_{i=1}^{3} (y_{i} \times \log(o_{outi})) + ((1-y_{i}) \times \log((1-o_{outi}))))
$$

$$
\frac{\partial (cross entropy)}{\partial O_{out1}} =  -1 *  ((y_{1} * (1/O_{out1}) + (1-y_{1})* (1/(1-O_{out1}))
$$

<blockquote>
<strong>수학적인 기본이 너무 튼튼한 분들은 이 부분을 무시해도 좋다.</strong> 필자가 이 부분을 길게 설명하는 이유는 앞으로 계산과정이 복잡하여 그 과정 중에는 <strong>계산하는 의미</strong> 를 잊기쉽기 때문이다. 우리의 목적은 <strong>계산(이건 컴퓨터가 한다..)</strong>이 아니라 <strong>의미를 이해하는 것</strong>이므로.. 계산을 시작하기전에 한번 더 짚고 넘어가자.
<br><br>
 도함수의 역할?<br>
 원래의 함수를 미분해서 얻어낸 도함수는 같은 입력에 대해 원래 함수의 기울기를 출력한다.
<br><br>
 우리가 수정하고자하는 값(<strong>변수</strong>)은?<br> Weight와 bias
<br><br>
 우리는 앞으로 예측값(y_pred)과 실제값(y_true)의 차이를 보여주는 <strong>오차 함수</strong>를 <strong>위의 변수들</strong>로 <strong>미분</strong>하여, 매 Step에서 그 <strong>변수</strong>들이 크고 작아짐에 따라 끼치는 영향(<strong>기울기</strong>)을 보며 <strong>값을 수정</strong>해나갈 것이다.
<br><br>
 이제 계산을 해보자!
</blockquote>

<h4 id='layer1' href='#layer1'>Back propagating the error (output Layer -> K Layer2)</h4>
<div style='float:left; width:40%; margin-right: 10px;'>
  <img src="/images/danial/back-prop/o_k_layer.png">
</div>
<!-- <div style=''> -->

먼저 K layer와 output layer 사이의 W<sub>ko</sub>가 E<sub>total</sub>(오차값)에 끼치는 영향을 확인하고 E<sub>total</sub>이 낮아지는 방향으로 값을 수정해보자.
이를 수식으로 표현해보면 아래와 같다.

$$
{E_{total}} = {(-1 * ((y * \log(O_{out}) + (1-y) * \log((1-O_{out}))}
$$

$$
{E_{total}} =
\left[ \begin{array}{cccc}
E_{1} & E_{2}\\\end{array} \right]
$$

$$
\delta W_{ko} =\frac{\partial E_{total}}{\partial W_{ko}} = \left[ \begin{array}{cccc}
\frac{\partial E_{1}}{\partial W_{k1o1}} & \frac{\partial E_{2}}{\partial W_{k1o2}}\\
\frac{\partial E_{1}}{\partial W_{k2o1}} & \frac{\partial E_{2}}{\partial W_{k2o2}}\\
\frac{\partial E_{1}}{\partial W_{k3o1}} & \frac{\partial E_{2}}{\partial W_{k3o2}}\end{array} \right]
$$

행렬의 각 원소들의 계산법은 같으므로 하나만 살펴보게되면,
$$
\frac{\partial E_{1}}{\partial W_{k1o1}}
$$
이 체인룰에 의해서
$$
\frac{\partial E_{1}}{\partial W_{k1o1}} = \frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O
_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial W_{k1o1}}
$$
가 된다. 여기서 관찰할 점은 컴퓨터가 순방향 전파를 진행하면서 O<sub>out1</sub>, O<sub>in1</sub>들을 다 계산했었다는 점이다. 즉 이들은 역전파가 이루어지는 이 순간에 상수로 작용한다. 그렇게 보면 도함수에 해당 값들을 대입하기만 하면 되므로 계산량이 상대적으로 크지 않다는 것을 알 수 있다.
<br>자 이제 <a href='#derivatives'>역전파 계산에 사용될 도함수</a>에서 보여준 함수에 값을 대입하면 된다.
$$
\frac{\partial E_{1}}{\partial O_{out1}} =  -1  * ((y_{1} * (1/O_{out1}) + (1-y_{1}) * (1/(1-O_{out1}))
$$
<blockquote>여기서 y1, y2는 실제 값을 의미한다.</blockquote>

$$
\frac{\partial O_{out1}}{\partial O_{in1}}  =
\mathrm{e}^{O_{in1}} * \mathrm{e}^{O_{in2}}/ (\mathrm{e}^{O_{in1}}+\mathrm{e}^{O_{in2}})^2
$$

이제 가장 마지막 함수를 살펴보자.

$$
\definecolor{first}{RGB}{245, 166, 35}
\definecolor{second}{RGB}{74, 144, 226}
\definecolor{third}{RGB}{189, 16, 224}
\frac{\partial O_{in1}}{\partial W_{k1o1}} = \frac{\partial (\textcolor{first}{W_{k1o1}} \times K_{out1} +
\textcolor{second}{W_{k2o1}} \times K_{out2} +
\textcolor{third}{W_{k3o1}} \times K_{out3} + b_{o1})}{\partial \textcolor{first}{W_{k1o1}}}
$$

이 부분의 미분한 값은 K<sub>out1</sub>이 된다.
즉,
$$
\frac{\partial O_{in1}}{\partial W_{k1o1}} = K_{out1}
$$

이제 종합해보면
$$
\delta W_{k1o1} = \frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O
_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial W_{k1o1}}
$$
이 된다. 이제 전체를 보게되면
$$
\delta W_{ko} = \left[ \begin{array}{cccc}
\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O
_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial W_{k1o1}} & \frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O
_{out2}}{\partial O_{in2}} * \frac{\partial O_{in2}}{\partial W_{k1o2}} \\
\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O
_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial W_{k2o1}}& \frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O
_{out2}}{\partial O_{in2}} * \frac{\partial O_{in2}}{\partial W_{k2o2}} \\
\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O
_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial W_{k3o1}} & \frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O
_{out2}}{\partial O_{in2}} * \frac{\partial O_{in2}}{\partial W_{k3o2}} \\ \end{array} \right]
$$
가 된다.

이제 W<sub>ko</sub>을 앞서 계산한 값에 맞춰서 수정하자.
$$
\acute{W_{kl}} = \left[ \begin{array}{cccc}
W_{k1o1} - (lr*\delta W_{k1o1}) & W_{k1o2} - (lr * \delta W_{k1o2}) \\
W_{k2o1} - (lr* \delta W_{k2o1}) & W_{k2o2} - (lr * \delta W_{k2o2}) \\
W_{k3o1} - (lr * \delta W_{k3o1}) & W_{k3o2} - (lr * \delta W_{k3o2}) \\ \end{array} \right]
$$
여기서 lr은 하이퍼 패러미터인 learning rate을 의미한다.

<h4 id='layer2' href='#layer2'>Back propagating the error (K Layer -> J Layer1)</h4>
<div style='float:left; width:40%; margin-right: 10px;'>
  <img src="/images/danial/back-prop/k_j_layer.png">
</div>

이제 다음 Layer를 확인해보며 조금 더 이해를 높여보자.
역시 이번에도 체인룰에 의해
$$
\frac{\partial E_{total}}{\partial W_{j1k1}} = \frac{\partial E_{total}}{\partial K_{out1}} * \frac{\partial K_{out1}}{\partial K_{in1}} * \frac{\partial K_{in1}}{\partial W_{j1k1}}
$$
가 된다. 이번엔 제일 우측부터 확인해보자.

$$
\definecolor{first}{RGB}{10, 233, 134}
\definecolor{second}{RGB}{74, 144, 226}
\definecolor{third}{RGB}{245, 166, 35}
\frac{\partial K_{in1}}{\partial W_{j1k1}} =
\frac{\partial(\textcolor{first}{W_{j1k1}} \times J_{out1} +
\textcolor{second}{W_{j2k1}} \times J_{out2} +
\textcolor{third}{W_{j3k1}} \times J_{out3} + b_{k1})}{\partial W_{j1k1}}
$$

위에서와 마찬가지로 이 부분은 한단계 전 Layer의 output을 의미한다. 즉,
$$
\frac{\partial K_{in1}}{\partial W_{j1k1}} = J_{out1}
$$
이 된다.
이제 그 다음이 중요하다.
$$
\frac{\partial K_{out1}}{\partial K_{in1}} = Sigmoid(K_{in1}) * (1 - Sigmoid(K_{in1}))
$$
항상 체인룰에 의해 곱해지는 이 미분 값이 바로 <strong id='v-g-p'>그래디언트 소멸 문제(Vanishing gradient problem)</strong>을 발생시키는 원인이기 때문이다.

그 이유는 아래 그림을 보면 명백하다.
<img src="/images/danial/back-prop/sigmoid_derivative.png">
1보다 작고, 가장 클 때 0.25인 Sigmoid 함수 미분값(기울기)이 매번 곱해지게 되면서 결국
$$
\delta W_{j1k1}
$$
를 0에 가깝게 만들게되고, 그 결과로
$$
\acute{W_{j1k1}} = W_{j1k1} - (lr * \delta W_{j1k1})
$$
식에 의해서 수정되는 Weight 값이 거의 변동이 없어지기 때문이다.

<strong>이 현상은 Layer가 많아질수록 심각해진다.</strong> 그 이유를 계속해서 계산하며 확인해보자!
$$
\frac{\partial E_{total}}{\partial K_{out1}}
$$
바로 이 녀석이 문제의 녀석이다.
위 식의 의미는
$$
\frac{\partial E_{total}}{\partial K_{out1}} = \frac{\partial E_{1}}{\partial K_{out1}} + \frac{\partial E_{2}}{\partial K_{out1}}
$$
체인룰로 풀게되면 아래와 같다.
$$
\frac{\partial E_{1}}{\partial K_{out1}} = \frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial K_{out1}}
$$

$$
\frac{\partial E_{2}}{\partial K_{out1}} = \frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}} * \frac{\partial O_{in2}}{\partial K_{out1}}
$$

위에서 계산한 Output layer -> K layer 부분 역전파에서 우린 이미
$$
\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}}
$$

$$
\frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}}
$$
는 계산했던 것이다. 그러니 그냥 그 값들을 대입하면 끝이다.

여기서! 위에서 말한 Layer 층이 많아질수록 소멸현상이 심해지는 이유는 이번 K-J레이어 간의 Weight를 업데이트 시키기 위해서 찾고있는
$$
\delta W_{j1k1}
$$
값은
$$
\frac{\partial K_{out1}}{\partial K_{in1}}, \frac{\partial O_{out1}}{\partial O_{in1}}
$$
두 개의 활성화 함수의 기울기 값이 곱해지고 있는 것이다.

<strong>물론 위의 예는 output Layer를 포함하고 있어서 아까전에 말한 Sigmoid 함수의 문제를 증폭시키는 것은 아니지만, 중앙에 위치한 어떤 Layer의 Weight값을 수정하기위한 계산이라고 생각해보라. 역전파가 Input에 가까워질수록 변경하게될 Weight값을 사실상 0이나 다름없게 되는 것이다.</strong>

이 부분이 이해가 되면 이번 포스트는 70%정도 역할을 해냈다고 본다.. (설명이 마음에 들었다면 부디 공유를 부탁!!)

다음도 중요한 부분이니 계속해서 살펴보자!

이제
$$
\frac{\partial E_{1}}{\partial K_{out1}} = \frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial K_{out1}}
$$
에서 남은 것은
$$
\frac{\partial O_{in1}}{\partial K_{out1}}
$$
뿐이다.

결국 이 식이 의미하는 것은 자세히 풀어서 살펴보면,
$$
\definecolor{first}{RGB}{245, 166, 35}
\definecolor{second}{RGB}{74, 144, 226}
\definecolor{third}{RGB}{189, 16, 224}
\frac{\partial O_{in1}}{\partial K_{out1}} = \frac{\partial (\textcolor{first}{W_{k1o1}} \times K_{out1} +
\textcolor{second}{W_{k2o1}} \times K_{out2} +
\textcolor{third}{W_{k3o1}} \times K_{out3} + b_{o1})}{\partial K_{out1}}
$$
이므로,
$$
\definecolor{first}{RGB}{245, 166, 35}
\textcolor{first}{W_{k1o1}}
$$
를 의미한다.

이제 통합해보면
$$
\left[ \begin{array}{cccc}
\frac{\partial E_{total}}{\partial K_{out1}}  \\
\frac{\partial E_{total}}{\partial K_{out2}}   \\
\frac{\partial E_{total}}{\partial K_{out3}} \\ \end{array} \right] =  \left[ \begin{array}{cccc}
(\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * W_{k1o1}) + (\frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}} * W_{k1o2})\\
(\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * W_{k2o1}) + (\frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}} * W_{k2o2})\\
(\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * W_{k3o1}) + (\frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}} * W_{k3o2})\\
\end{array} \right]
$$
가 되겠다!

여기서 <strong>포인트</strong>는 <strong id='initial-weight'>Weight 초기값 설정</strong>이 너무나도 중요하다는 것이다.

<strong>역전파에서 계속해서 곱해지는 값 중 하나가 바로 이전 Layer(역전파 기준으로 이전)의 Weight값이라는 점.
곱셈으로 연결되어 있는 체인룰에 의해서 초기값을 0으로 하게되면 당연히 업데이트가 일어나지 않게 되는 것이다. 인공지능분야 유명한 교수인 Hinton교수님이 하신 말씀이 We initialized the weights in a stupid way. 즉! 초기값을 어떻게 설정하는지가 매우 중요하다는 것! 이 부분을 꼭 기억하자.
</strong>

이제 위에서 구한 모든 식을 곱하게 되면
$$
\delta W_{jk} =  \left[ \begin{array}{cccc}
\frac{\partial E_{total}}{\partial K_{out1}} * \frac{\partial K_{out1}}{\partial K_{in1}} * \frac{\partial K_{in1}}{\partial W_{j1k1}} & \frac{\partial E_{total}}{\partial K_{out2}} * \frac{\partial K
_{out2}}{\partial K_{in2}} * \frac{\partial K_{in2}}{\partial W_{j1k2}}& \frac{\partial E_{total}}{\partial K_{out3}} * \frac{\partial K
_{out3}}{\partial K_{in3}} * \frac{\partial K_{in3}}{\partial W_{j1k3}} \\
\frac{\partial E_{total}}{\partial K_{out1}} * \frac{\partial K
_{out1}}{\partial K_{in1}} * \frac{\partial K_{in1}}{\partial W_{j2k1}}& \frac{\partial E_{total}}{\partial K_{out2}} * \frac{\partial K
_{out2}}{\partial K_{in2}} * \frac{\partial K_{in2}}{\partial W_{j2k2}} & \frac{\partial E_{total}}{\partial K_{out3}} * \frac{\partial K
_{out3}}{\partial K_{in3}} * \frac{\partial K_{in3}}{\partial W_{j2k3}} \\
\frac{\partial E_{total}}{\partial K_{out1}} * \frac{\partial K
_{out1}}{\partial K_{in1}} * \frac{\partial K_{in1}}{\partial W_{j3k1}} & \frac{\partial E_{total}}{\partial K_{out2}} * \frac{\partial K
_{out2}}{\partial K_{in2}} * \frac{\partial K_{in2}}{\partial W_{j3k2}} & \frac{\partial E_{total}}{\partial K_{out3}} * \frac{\partial K
_{out3}}{\partial K_{in3}} * \frac{\partial K_{in3}}{\partial W_{j3k3}} \\ \end{array} \right]
$$
를 구할 수 있다.

역시 마찬가지로,
$$
\acute{W_{jk}} = \left[ \begin{array}{cccc}
W_{j1k1} - (lr*\delta W_{j1k1}) & W_{j1k2} - (lr * \delta W_{j1k2}) &W_{j1k3} - (lr * \delta W_{j1k3}) \\
W_{j2k1} - (lr* \delta W_{j2k1}) & W_{j2k2} - (lr * \delta W_{j2k2}) &W_{j2k3} - (lr * \delta W_{j2k3}) \\
W_{j3k1} - (lr * \delta W_{j3k1}) & W_{j3k2} - (lr * \delta W_{j3k2}) & W_{j3k3} - (lr * \delta W_{j3k3}) \\ \end{array} \right]
$$
방식으로 수정해주면 되겠다.

<br><br>
아직 J layer에서 Input layer 역전파가 남은 것은 알지만.. 사실상 계산법이 차이가 크게 없기 때문에 더 설명하지는 않고 계산은 여기서 끝내도록 하겠다. 여기까지 읽으신 분에게 박수를.. 보낸다. 이제 위에서 설명한 소멸 문제와 초기값에 대해서 좀 더 얘기하고 이번 포스트를 마무리하도록 하겠다.

<h3 id='conclusion'>문제점 해결 방법 및 결론</h3>

첫 번째는 활성화 함수 Sigmoid가 발생시키는 <strong>그래디언트 소멸 문제(Vanishing gradient problem)</strong>는 활성화 함수 종류를 바꾸면 해결된다(물론 무조건 좋은 활성화 함수란 없다고 한다.. 수렴이 잘되는 것을 찾는 것이 우리의 역할). 그 중에 하나는 Relu인데, 특징은
$$
Relu = max(0, x)
$$
라서 음수는 다 0으로 만들고 양수인 경우만 남기게 된다. 덕분에 기울기도 0이나 1만 나오게되서 계산이 편하다. 그 외에도 하이퍼탄젠트 활성화 함수 등이 있으니 따로 검색해보길 추천한다. 혹은 [데이터 사이언스 스쿨 - 신경망 성능 개선](https://datascienceschool.net/view-notebook/f18248a467e94c6483783afc93d08af9/)에서도 확인이 가능하다.

가중치 초기화 관련된 문제도 위의 링크에서 확인할 수 있는데, TensorFlow에서 Xavier가 2010년에 제시한 Xavier initializer를 사용할 수 있는데 대회에서 수상한 경우에는 주로 이 Weight 초기화법을 사용했다고 하니 기억하면 좋을 것으로 보인다.

사용법은 아래처럼 사용하면 된다.

```python
import tensorflow as tf
tf.contrib.layers.xavier_initializer()
```

드디어 이 포스팅을 완료했다.. 만약 도움이 많이 되었다면 꼭 공유를 해주면 감사하겠다!!

### Related Posts
