---
title: <Neural Network> 인공신경망에 대한 이해(Part 2 - Back Propagation)
categories:
  - Danial Nam
tags:
  - Deep Learning
  - Neural Network
  - Artificial Intelligence
date: 2018-11-19 00:24:13
thumbnail:
---

<div style='display: none;'>
<img src="/images/danial/back-prop/thumbnail.png">
</div>

<br>

이번 포스트(Part 2)에서는 인공 신경망을 가능하게 한 **Back Propagation** 에 대해 알아보도록 하겠다.(모바일에는 최적화되어있지 않으니 가능하면 PC로 보시길 추천한다)

만약 인공 신경망의 기본 개념과 **Feedforward Propagation** 에 대해서 잘 모른다면 [이전 포스트(Part 1)](https://Databuzz-team.github.io/2018/11/05/Back-Propagation/)를 먼저 보고 오기 바란다.

### 목차
- <a href='#why-back-propagation'>왜 Back propagation를 이해해야 할까?</a>
- <a href='#back-propagation'>Back Propagation 설명</a>

<h3 id='why-back-propagation' href='#why-back-propagation'>왜 Back propagation를 이해해야 할까?</h3>

"어차피 TensorFlow를 사용하면 다 자동으로 계산해주는 것인데 왜 우리가 공부해야 하는 것일까?"

합리적인 질문이다. 공식만 봐도 어려워보이는 이 부분을 공부하는 것이 동기부여가 쉽게 되지않는것이 사실이기 때문에..

하지만, [Yes you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b) 글에서 설명했듯 이 **Back Propagation** 은 **Leaky Abstraction** 라는 것이다.

> **Leaky Abstraction**<br>
Joel Spolsky이 설명한 [The Law of Leaky Abstraction](https://www.joelonsoftware.com/2002/11/11/the-law-of-leaky-abstractions/)에서 사용된 표현으로써 프로그래밍 언어를 추상화시켜서 내부 구현을 모르도록 만들어놨지만, <strong style='color:red'>결국 제대로 사용하려면 내부 구현을 상당 부분 알아야 한다는 것을 의미한다.</strong>

이 포스트에서는 인공 신경망의 문제점에 대해서는 다루지 않기 때문에 왜 Leaky Abstraction이라 설명했는지 궁금한 경우에는 위의 블로그 링크를 참고하길 바란다.

<h2 style='color: red;'>이 포스트는 아직 작성 중에 있습니다. 최대한 빨리 작성해서 올리도록 하겠습니다!</h2>

### Related Posts
