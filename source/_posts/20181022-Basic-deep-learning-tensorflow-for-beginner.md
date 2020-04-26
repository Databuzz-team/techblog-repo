---
title: <Deep Learning> An introduction to Deep Learning with Tensorflow(Part-1)
categories:
  - Danial Nam
tags:
  - Machine Learning
  - Artificial Intelligence
  - Deep Learning
  - Tensorflow
date: 2018-10-22 23:27:01
thumbnail:
---
<div class="danthetech-intro-wrap">
  <a class="danthetech-intro-a" href="https://danthetech.netlify.app/DataScience/basic-concept-of-tensorflow/">
    <img class="danthetech-img-wrap" src="https://cdn-images-1.medium.com/max/1600/1*7lklTJQytHz8w7Eeqz5ZhA.png">
    <div class="danthetech-p-wrap">
      <h1 class="danthetech-intro-title">
        An introduction to Deep Learning with Tensorflow(Part-1)
      </h1>
      <p class="danthetech-intro-p">
        <span class="danthetech-intro-strong">이 컨텐츠는 DanTheTech 블로그로 옮겨졌습니다!</span>
        불편을 끼쳐드려 죄송합니다. 이 링크를 클릭하셔서 확인해주시면 정말 감사하겠습니다.
        앞으로도 DataScience, App Development부터 DevOps관련 자료 등 도움될만한 글이 많이 올릴 예정이니 자주 들려주세요! :)
      </p>
    </div>
  </a>
</div>

<br>

최근에 번역한 ["DATA SCIENTISTS에게 가장 요구되는 기술(SKILLS)들"](https://databuzz-team.github.io/2018/10/15/The-Most-in-Demand-Skills-for-Data-Scientists/) 글에서 확인했듯, 급부상하는 분야인 Machine Learning에서도 Deep Learning은 우리가 앞으로 이 분야에서 일하자면 꼭 공부해야 한다.

먼저 간단하게 **Machine Learning** 에 대해서 알아보자.

### Machine Learning?
사실 역사는 이미 오래된 분야다. 하지만 최근들어서야 급부상하는 이유는 무엇보다도 Computing Power의 성장과 방대한 데이터가 있다는 점이다.

크게는 지도학습과 비지도학습으로 나뉘는데, 지도학습은 Input과 Output을 알려주고 그 사이에 존재하는 로직을 기계가 학습하도록 하는 것이며, 비지도학습은 Input만 알려주고 데이터가 가진 특징 속에서 스스로 학습해나가는 것을 말한다.

아래의 영상은 최근에 공개된 영상인데 Deep Learning의 현재 수준이 이미 놀라운 수준임을 알 수 있다.

{% iframe https://www.youtube.com/embed/PCBTZh41Ris 560 315%}

만약 가장 트렌디한 머신러닝 기술들을 보고싶다면 [링크](https://arxiv.org/)를 확인하는 것이 가장 좋다고한다. 학회에 제출하고 논문이 Accept가 되는데 평균적으로 7개월이 걸리는데, 그 기간이면 이미 새로운 기술이 나오는 상황이라.. 만약 트렌드한 기술을 캐치하자면 꼭 [링크](https://arxiv.org/)에서 확인하자.

### 왜 Tensorflow인가?
Tensorflow는 Google이 개발한 Library인데, 현재는 가장 사랑을 받고 있는 Library로써, Google 자체에서도 Google Photo, Google Voice Search 등은 모두 Tensorflow를 간접적으로 사용하고 있는 App 들이다.

### Tensorflow 기본 개념
> 이 부분은 [A beginner introduction to TensorFlow (Part-1)](https://towardsdatascience.com/a-beginner-introduction-to-tensorflow-part-1-6d139e038278)을 번역 정리하였다.

- **Tensorflow** 의 Core는 그래프(computational graph)와 Tensor로 이루어져있다.
- **Tensor와 Vector의 차이점** 은 Tensor는 크기만 가진경우도 존재한다는 점이다. 즉 Vector는 Tensor의 특수상황이며, 부분집합으로 볼 수 있다.
- **Tensor** 는 이해했으니 **Flow** 를 살펴보자. **Flow** 는 computational graph 혹은 단순한 graph라고 볼 수 있다. cyclic한 구조는 아니며, 각 노드(아래 그림에서 동그라미)는 덧셈, 뺄셈 등의 기능을 가지고 있다. <div>
<img src="https://cdn-images-1.medium.com/max/1600/1*7lklTJQytHz8w7Eeqz5ZhA.png"/>
<span style='font-size:12px; text-align:center; display:block; color: #999;'><a href='https://towardsdatascience.com/a-beginner-introduction-to-tensorflow-part-1-6d139e038278'>출처 : A beginner introduction to TensorFlow (Part-1)</a></span>
</div>

- **e = (a+b) * (b+1)**
  기능(operation)적인 역할을 해야 하는 모든 연결되는 노드(꼭짓점)은 graph의 시작일 수는 없고, Tensor를 받거나 새로운 Tensor를 생성하는 역할을 한다. 또한, computational graph는 항상 복잡한 계층 구조로 되어있는데, 위의 그림에서도 마찬가지로 표현되어 있듯 a+b는 c로 b+1은 d로 표현될 수 있다.
- **e = (c)*(d), c = a+b and b = b+1**
  위의 그림에서 명백하게 표현되어있듯 각 노드는 전 노드에 의존적이어서 c는 a, b 없이 나올 수 없고, e는 c, d 없이 나올 수 없다. 단 c, d처럼 같은 계층에 존재하는 노드들은 상호 독립적이다. 이 점은 computational graph를 이해할 때 가장 중요한 부분으로써, **같은 레벨에 있는 노드 c의 경우는 d가 먼저 계산되어야 할 이유가 없고, 평행적으로 실행될 수 있다.**

- 위에서 설명한 computational graph의 평행 관계(parallelism)가 가장 중요한 개념이니 꼭 숙지해야 한다. 이 평행 관계의 의미는 c 계산이 끝나지 않았다고, d 계산은 평행적으로 이루어진다는 점이며, tensorflow는 이 부분을 멋지게 해낸다.

### Tensorflow 분할 실행
<div>
<img src="https://cdn-images-1.medium.com/max/1600/1*cok4bMhTvE93UdGmRblEyw.png"/>
<span style='font-size:12px; text-align:center; display:block; color: #999;'><a href='https://towardsdatascience.com/a-beginner-introduction-to-tensorflow-part-1-6d139e038278'>출처 : A beginner introduction to TensorFlow (Part-1)</a></span>
</div>

- **Tensorflow** 는 여러 기계에 평행적으로 계산을 실행하여 훨씬 빠른 연산을 할 수가 있는데, 따로 설정할 필요 없이 내부적으로 설정이 된다.
<br>
  > 위의 그림에서 왼쪽은 single Tensorflow session을 사용한 경우라서 single worker가 존재하는 것이고, 오른쪽은 multiple workers를 사용한 경우


- Worker들은 서로 다른 기기에서 독립적으로 연산을 하고 다음 노드에 해당하는 Worker에게 Result를 넘겨준다. 이때, Delay로 인한 성능 저하가 일어날 수 있는데 이는 주로 **Tensor** 의 **Size** 에서 발생하므로 어떤 **자료형** 을 설정할 것인지가 **중요한 문제** 다.

### 결론
이번 Part1 글에서는 Machine Learning과 Tensorflow에 대한 소개를 중점으로 썼고, Part2에서는 Tensorflow의 기본 문법과 MNIST Digit 이미지 분류하는 코딩에 대해서 포스팅하겠다.

---
### Related Posts
[A beginner introduction to TensorFlow (Part-1)](https://towardsdatascience.com/a-beginner-introduction-to-tensorflow-part-1-6d139e038278)
