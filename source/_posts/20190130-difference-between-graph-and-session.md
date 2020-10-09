---
title: <Tensorflow> Tensorflow에서 Graph와 Session의 차이
categories:
  - Danial Nam
tags:
  - Tensorflow
  - Deep Learning
  - Graph
  - Session
date: 2019-01-30 01:07:47
thumbnail:
---
<div style='display: none;'>
<img src="/images/danial/tensorflow.jpeg">
</div>

### About
Tensorflow로 이것저것 도전하면서
```python
ValueError: Operation name: "init"
op: "NoOp"
input: "^val/Assign"
 is not an element of this graph.
```
오류라던가
```python
Tensor Tensor("val:0", shape=(), dtype=int32_ref) is not an element of this graph.
```
와 같은 오류들을 마주치면서 graph에 대해서 좀 더 깊게 이해하는 것이 좋겠다는 생각이 들었다.

그래서 찾던 도중에 좋은 예제가 있어서 이번 포스팅을 통해서 소개하고자한다.

### Graph?
1. Graph는 operation을 정의한 것을 말한다. Tensor와 Node간의 계산 등을 포함한 설계도를 의미하며 Data와는 관계가 없다.
2. Graph를 Session을 통해서 실행할 때, 우리가 layer간에 설치한 dimensions나 type등을 체크하며 문제가 있을시에는 예외를 발생시킨다.
3. Graph를 실행하기 위해서는 Data를 feed 해야한다.

### Session?
Graph에게 전화를 거는 것이다. 특정 Graph에게 전화를 걸고 우리가 원하는 결과를 얻기위해 일을 시키는 것이라고 생각하면 된다.
```python
with tf.Session(graph=graph_we_want_to_connect) as sess:
  sess.run([op], feed_dict={data:data})
```

### 실습
위의 뜻을 해석해보면 다양한 모델(Graph)을 만들고 세션마다 원하는 모델에 연결해서 학습을 진행할 수 있다는 것이다.

무슨 뜻인지 아래의 예제를 통해서 확인해보자.

```python
import tensorflow as tf

graph_1 = tf.Graph()

with graph_1.as_default():
  W_1 = tf.Variable(100, name="graph_1_W")
  init_graph_1 = tf.global_variables_initializer()

graph_2 = tf.Graph()

with graph_2.as_default():
  W_2 = tf.Variable(200, name="graph_2_W")
  init_graph_2 = tf.global_variables_initializer()

with tf.Session(graph=graph_1) as sess:
  sess.run(init_graph_1)
  print(sess.run(W_1))

# output 100
```
이 경우에 아래처럼 실행하면
```python
with tf.Session(graph=graph_1) as sess:
  sess.run(init_graph_2)
  print(sess.run(W_1))
```
아래같은 오류가 난다.
```python
Operation name: "init"
op: "NoOp"
input: "^graph_2_W/Assign"
 is not an element of this graph.
```
오류 의미 그대로이다. 우리가 실행하고자 한 init_graph_2가 graph_1에 해당하는 것이 아니어서 오류가 발생한 것이다.

변수의 경우도 마찬가지로 아래처럼 실행하면 오류가 발생한다.
```python
with tf.Session(graph=graph_1) as sess:
  sess.run(init_graph_1)
  print(sess.run(W_2))
```

```python
Tensor("graph_2_W:0", shape=(), dtype=int32_ref) is not an element of this graph.
```

### 결론
Graph는 모델이며, Session은 원하는 모델을 선택하여 전화를 걸고 일을 시키는 행위를 말한다.

### Related Posts
[<Tensorflow> Tensorflow에서 Graph와 Session의 차이](https://danthetech.netlify.com/DataScience/tensorflow-difference-between-graph-and-session/)
[Understanding Session and Graph](http://goingmyway.cn/2017/07/14/Understanding-Session-and-Graph/)
