---
title: <Deep Learning> An introduction to Deep Learning with Tensorflow(Part-2)
categories:
  - Danial Nam
tags:
  - Machine Learning
  - Artificial Intelligence
  - Deep Learning
  - Tensorflow
date: 2018-10-24 10:03:01
thumbnail:
---

이번 포스트에서는 Tensorflow의 기본 문법에 대해서 알아보자.(Tensorflow설치법에 대해서는 다루지 않을 것)

### 1. Dataflow
<div style='width:40%; float:left;'>
<img src='https://www.tensorflow.org/images/tensors_flowing.gif' />
<p style='text-align:center'><a href='https://www.tensorflow.org/guide/graphs'>출처 :Graphs and Sessions</a></p>
</div>

<div style='width: 60%; float:right; color: #333;'>
<a href='https://databuzz-team.github.io/2018/10/22/Basic-deep-learning-tensorflow-for-beginner/'>이전 포스트 Part 1</a>에서 이미 다룬 내용이며 개념 설명이라 지루할 수는 있지만, Tensorflow를 사용하기 위해선 꼭 이해해야 하는 부분을 다시 그림과 함께 짧게 설명하고자 하니 꼭 읽어주시길 바란다.

옆의 그림은 <strong>Tensorflow</strong> 공식 홈페이지에 가면 나와있는 이미지 파일인데, 정말 직관적으로 Tensorflow의 <strong>Dataflow Graph(Node And Operation)</strong> 를 표현해냈다.

먼저 <strong>Tensor</strong> 는 옆에서 보이는 검은 라인이고(영어로는 Edge이지만, 라인이 내겐 더 직관적이라 그렇게 설명한 것), <strong>Operation</strong> 은 노드들, 그림에서 타원들을 의미한다. 즉, Tensor가 Operation으로 들어가서 해당 Operation에서 설정한 연산을 진행하고 다시 Tensor를 Output으로 내보내는 것이다.
<br>
<br>
<blockquote>필자가 이해한 바대로라면 Tensor나 Operation이라는 낯선 단어들을 사용해서 어렵게 느껴지지만 결국은 함수의 기능을 한다고 봐주면 되겠다. 차이점은 Graph는 선언이고 Session을 통해서 Run을 한다는 것</blockquote>
<br>
<br>
물론! 끝은 Output으로 값을 내보내는 것을 목적으로 하는 것은 아니다. 우리의 목적은 <strong>W 가중치</strong> 를 <strong>Update</strong> 하는 것이므로, 마지막에 우리가 <strong>Optimizer</strong> 의 변수로 설정한 <strong>W1, b1, W2, b2 ...</strong> 들이 <strong>Update</strong> 되는 것으로 <strong>Session.run()</strong> 이 종료된다.
```python
import tensorflow as tf
sess = tf.Session()
sess.run(task)
sess.close()
```

</div>

<div style='width: 100%; clear:both; color: #333;'>
<h3>2. Operation의 name과 scope의 간략한 소개</h3>
본격적으로 코드에 대해 설명하기 전에 <strong>Debugging</strong> 에 도움이 되는 정보인 Operation name에 대해서 간략하게 살펴만 보자.
```python
c_0 = tf.constant(0, name="c")  # => operation 이름은 "c"

# 이미 사용된 이름은 자동으로 유니크화 시킨다.
c_1 = tf.constant(2, name="c")  # => operation 이름은 "c_1"

# Name scope는 접두사로 붙게되는데 나중에 설명할 Tensorboard에서 확인할 때 훨씬 편리하다.
with tf.name_scope("outer"):
  c_2 = tf.constant(2, name="c")  # => operation 이름은 "outer/c"

  # Name scope 아래로는 경로로 계층을 표현한다.
  with tf.name_scope("inner"):
    c_3 = tf.constant(3, name="c")  # => operation 이름은 "outer/inner/c"

  c_4 = tf.constant(4, name="c")  # => operation 이름은 "outer/c_1"

  with tf.name_scope("inner"):
    c_5 = tf.constant(5, name="c")  # => operation 이름은 "outer/inner_1/c"
```
<p style='text-align:center; margin:0'><a href='https://www.tensorflow.org/guide/graphs'>출처 :Graphs and Sessions</a></p>

</div>

<div style='width: 100%; clear:both; color: #333;'>
<h3>3. Tensorflow 가장 많이 사용되는 함수들을 알아보자</h3>이 포스트에서는 필자가 많이 사용된다고 생각하는 가장 기본적인 함수들만 작성했는데, 이 외에도 거의 모든 수학 연산은 다 구현되어 있으니 자세한 API는 <a href='https://www.tensorflow.org/api_docs/python/'>링크</a>를 통해서 찾아보도록 하자.
<ul><li><h5>tf.placeholder()</h5>
```python
tf.placeholder(
    dtype,
    shape=None,
    name=None
)
```
<strong>tf.placeholder()</strong> 는 머신러닝에서 무조건 사용하는 함수이며, 구조는 위의 코드에서 보듯 dtype, shape, name으로 이루어져있는데 핵심은 <strong>shape</strong> 이다.
<br>
<br>
<blockquote>다른 포스트에서 언급하겠지만 tensorflow에서 shape을 이해하는 것은 매우 중요하다.</blockquote>
<br>
예를 들어 집값을 예측하는 모델을 우리가 만들고 있고, 집의 <strong>Feature(특징)</strong> 는 <strong>rooms, is_riverside</strong> 로 이루어져 있다고 하자. 그리고 만약 우리가 최종적으로 사용할 <strong>Feature</strong> 의 수는 위에서 말한 두 가지면 충분하다고 결정했다고 본다면, <strong>Input</strong> 데이터의 <strong>shape</strong> 으로 <strong>[?, 2]</strong>, 즉 컬럼은 2 개로 결정을 한 상태라는 것이다. 하지만, 집의 개수, 즉 데이터의 개수는 많을수록 좋은 것이므로 변동의 여지가 언제나 있는 값일 뿐 아니라 최종적으로 우리가 새로운 데이터를 예측하려 할 때에도 변하는 값이라는 것이다.
<br>
<br>
그런 이유에서 <strong>tensorflow</strong> 에서는 <strong>placeholder</strong> 라는 함수를 제공하는 것이며, <strong>Feature(X)</strong> 와 <strong>Label(y)</strong> 은 <strong>placeholder</strong> 를 사용해서 넣어준다. 주의할 점은 <strong>sess.run()</strong> 시에 <strong>feed_dict</strong> 에 꼭 값을 직접 <strong>넣어(feed)</strong> 주어야 한다는 것.

```python
import tensorflow as tf
x_data = np.array([[3, 1], [4, 0], [5, 1]])
y_data = np.array([[120000], [100000], [200000]])

X = tf.placeholder(tf.float32, shape=[None, 2], name="X")
# 이 예제에서는 각 집마다의 가격을 예측하는 것이므로, shape은 [None, 1]이 된 것.
y = tf.placeholder(tf.float32, shape=[None, 1], name="y")
```
</li>
<li><h5>tf.Variable()</h5>
```python
tf.Variable(<initial-value>, name=<optional-name>)
```
머신러닝을 통해서 구하고자하는 값인 <strong>Weight</strong> 나 <strong>Bias</strong> 와 같은 값은 <strong>tensorflow</strong> 의 <strong>tf.Variable()</strong> 함수를 사용해서 선언해야한다. 구조는 위의 code에서 보듯 아주 단순하며, 보통은 Random하게 초기화하는 경우가 많으므로 행렬곱을 할 상대인 <strong>X</strong> 와 예측 값으로 내보내는 <strong>y</strong> 의 <strong>shape</strong> 을 고려해서 <strong>tf.random_normal()</strong> 을 사용하게 된다.
```python
W = tf.Variable(tf.random_normal([2, 1]), name='wight')
b = tf.Variable(tf.random_normal([1]), name='bias')
```
</li>
<li><h5>tf.matmul()</h5>
```python
tf.matmul(
    a,
    b,
    transpose_a=False,
    transpose_b=False,
    adjoint_a=False,
    adjoint_b=False,
    a_is_sparse=False,
    b_is_sparse=False,
    name=None
)
```
머신러닝에서는 <strong>원소간의 곱(Element-wise multiplication)</strong> 보다는 <strong>행렬곱(Matrix multiplication)</strong> 이 훨씬 많이 쓰이므로 <strong>tf.matmul()</strong> 은 꼭 알아야하는 함수이다.
```python
hypothesis = tf.matmul(X, W) + b
```
</li>

<li><h5>tf.train module</h5>
오늘 소개할 마지막은 함수가 아닌 <strong>모듈(Module)</strong> 이다. 아래는 가장 보편적인 <strong>Optimizer</strong> 인 <strong>GradientDescentOptimizer</strong> 로 예를 들었지만, 훨씬 많은 모델들을 <strong>tensorflow</strong> 에서는 제공하고 있으니, 이 외에 필요한 정보는 <a href='https://www.tensorflow.org/api_docs/python/tf/train'>링크</a>에서 확인하도록 하자.
<br>
<br>
<blockquote><strong>목표 함수(Cost function)</strong> 에 대해서 이 포스트에서는 특별히 다루지 않지만, 다음 포스트들에서 CNN, RNN 등의 알고리즘을 구현하며 설명을 추가하겠다.
</blockquote>
```python
cost = tf.reduce_mean(tf.square(hypothesis - y))
train = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)
```
</li>
</ul>

<h3>4. 마무리</h3>

이번 포스트는 어려운 내용이 없지만 <strong>tensorflow</strong> 를 공부하며 간단한 모델들을 구현해보며 가장 자주 사용되고 중요하다고 느꼈던 점을 정리해보았는데, 처음 시작하는 사람들에게 꼭 도움이 되길 바란다.
<br>
<br>
아래는 가장 간단하게 회귀분석 모델을 구현한 코드이며, 위에서 설명한 개념들을 정말 간단한 예제이긴 하지만, 대략적으로 어떻게 쓰이나 보여주기 위해서 작성해보았다.
```python
import tensorflow as tf
x_data = np.array([[3, 1], [4, 0], [5, 1]])
y_data = np.array([[120000], [100000], [200000]])

# hyper parameter
lr = 0.01
n_epoch = 2000

X = tf.placeholder(tf.float32, shape=[None, 2], name="X")
y = tf.placeholder(tf.float32, shape=[None, 1], name="y")

W = tf.Variable(tf.random_normal([2, 1]), name='wight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))
train = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

with tf.Session() as sess:
    # 변수가 있는 경우에는 초기화를 실행해줘야 한다.
    sess.run(tf.global_variables_initializer())
    # train이 반환하는 값은 우리에게 필요없다.
    for step in range(n_epoch):
        c, _ = sess.run([cost, train], feed_dict={X: x_data, y: y_data})
        if step % 500 == 0:
            print("Step :", step, "Cost :", c)
            # x, y를 임의로 만든거라..
            # 이 부분은 train data를 학습시키는지 확인하는 목적 외에는 없다.
            print(sess.run(hypothesis, feed_dict={X: x_data}))
```

</div>


### Related Posts
