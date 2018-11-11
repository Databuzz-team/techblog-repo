---
title: <Neural Network> 신경망 Feedforward & Back Propagation에 대한 이해
categories:
  - Danial Nam
tags:
  - Back Propagation
  - Deep Learning
  - Neural Network
  - Artificial Intelligence
date: 2018-11-05 14:16:52
thumbnail:
mathjax: true
---

<br>

이번 포스트는 TensorFlow로 DNN을 구현하기 전에 먼저 인공 신경망을 가능하게 한 **Back Propagation** 에 대한 이해를 도울 수 있는 [블로그 Back-Propagation is very simple. Who made it Complicated ? ](https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c?fbclid=IwAR2bn8rI7yxT5BC-0Ovy2IvzHEb2z_uaVdaw-uPLK-9-SmI2JoP6EK7do-0)내용을 번역하여 소개하고자 한다.

<div>
<img src='https://cdn-images-1.medium.com/max/2000/1*fnU_3MGmFp0LBIzRPx42-w.png'>
<a style='display: block; text-align: center;' href='https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c?fbclid=IwAR2bn8rI7yxT5BC-0Ovy2IvzHEb2z_uaVdaw-uPLK-9-SmI2JoP6EK7do-0'>출처 : [Medium] Back-Propagation is very simple. Who made it Complicated ?</a>
</div>

### 목차
- <a href='#nn-history'>Neural Network 역사 및 Back propagation의 중요성</a>
- <a href='#why-back-propagation'>왜 Back propagation를 이해해야 할까?</a>
- <a href='#back-propagation'>Back Propagation 설명</a>


<h3 id='nn-history' href='#nn-history'>Neural Network 역사</h3>

 1943년, 워런 맥컬록(Warren McCulloch)와 월터 피츠(Walter Pitts)의 수학과 임계 논리(Threshold logic)라 불리는 알고리즘을 바탕으로 신경망을 위한 계산한 모델이 만들어지며 신경망 연구의 초석을 닦으며 **Neural Network** 역사가 시작되었다.

 하지만, 1969년에 마빈 민스키(Marvin Minsky)와 시모어 페퍼트(Seymour Papert)에 의해 기계학습 논문이 발표된 후 침체되었는데, 그 이유는 두 가지였다.

1. 단층 신경망은 선형으로 분류하기 때문에 아래의 그림처럼 문제가 배타적 논리합 회로(XOR problem)인 경우에는 해결하지 못한다.
<div>
<img src='https://cdn-images-1.medium.com/max/2000/0*qdRb80zUpJPtrbRD.'/>
<a style='display: block; text-align: center;' href='https://medium.com/@jayeshbahire/the-xor-problem-in-neural-networks-50006411840b'>
출처 : [Medium] Back-Propagation is very simple. Who made it Complicated ?</a>
</div>

2. Computing power가 부족하다.

위의 두 가지 문제점 외에도 가중치를 업데이트하기 위한 회로 수학 계산이 너무 복잡하였기에 **오차역전파법(Back Propagation)** 이 세상에 나오기 전까지는 연구가 침체될 수밖에 없었다.

<h3 id='why-back-propagation' href='#why-back-propagation'>왜 Back propagation를 이해해야 할까?</h3>

"어차피 TensorFlow를 사용하면 다 자동으로 계산해주는 것인데 왜 우리가 공부해야 하는 것일까?"

합리적인 질문이다. 공식만 봐도 어려워보이는 이 부분을 공부하는 것이 동기부여가 쉽게 되지않는것이 사실이기 때문에..

하지만, [Yes you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b) 글에서 설명했듯 이 **Back Propagation** 은 **Leaky Abstraction** 라는 것이다.

> **Leaky Abstraction**<br>
Joel Spolsky이 설명한 [The Law of Leaky Abstraction](https://www.joelonsoftware.com/2002/11/11/the-law-of-leaky-abstractions/)에서 사용된 표현으로써 프로그래밍 언어를 추상화시켜서 내부 구현을 모르도록 만들어놨지만, <strong style='color:red'>결국 제대로 사용하려면 내부 구현을 상당 부분 알아야 한다는 것을 의미한다.</strong>

이 포스트에서는 인공 신경망의 문제점에 대해서는 다루지 않기 때문에 왜 Leaky Abstraction이라 설명했는지 궁금한 경우에는 위의 블로그 링크를 참고하길 바란다.

<h3 id='back-propagation' href='#back-propagation'>Back Propagation</h3>

### 네트워크 초기화
<img src="/images/danial/back-prop/network.png">

$$
Input = \left[ \begin{array}{cccc}
i_{1} \\ i_{2} \\\end{array} \right] W_{ji} = \left[ \begin{array}{cccc}
W_{j1i1} & W_{j1i2} \\
W_{j2i1} & W_{j2i2} \\
W_{j3i1} & W_{j3i2} \\ \end{array} \right]
W_{kj} = \left[ \begin{array}{cccc}
W_{k1j1} & W_{k1j2} & W_{k1j3} \\
W_{k2j1} & W_{k2j2} & W_{k2j3} \\
W_{k3j1} & W_{k3j2} & W_{k3j3} \\ \end{array} \right]
W_{ok} = \left[ \begin{array}{cccc}
W_{o1k1} & W_{o1k2} & W_{o1k3} \\
W_{o2k1} & W_{o2k2} & W_{o2k3} \\ \end{array} \right]
Output = \left[ \begin{array}{cccc}
o_{1} \\ o_{2} \\\end{array} \right]
$$

### <span style='color: red;'>현재 이 포스트는 아직 미완상태입니다. 최대한 빨리 완료해서 올리겠습니다.ㅠㅠ</span>

먼저 신경망의 계산 과정인 순방향 전파(Feedforward propagation)를 살펴보자.

### Layer 1 (Input -> J)



### Related Posts
[Backpropagation calculus | Deep learning, chapter 4 by 3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8&fbclid=IwAR2lsWOByt_MrzBkv5-Dc9P6JIdvHv1pUELE5q-0SVqQ73b6tS-RYGUI9eM)
[신경망 기초 이론](https://datascienceschool.net/view-notebook/0178802a219c4e6bb9b820b49bf57f91/)
