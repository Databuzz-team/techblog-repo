---
title: <MACHINE LEARNING> CHAPTER1. 한눈에 보는 머신 러닝
categories:
  - HyunGeun Yoon
tags:
  - Machine Learning
date: 2018-10-13 14:34:56
thumbnail:
---

Hands-On Machine Learning with Scikit-Learn & TensorFlow 책을 읽고 공부하면서 내용을 요약하고 정리한 것입니다.
<!-- more -->
### 1.1 머신 러닝이란?

> 어떤 작업 T에 대한 컴퓨터 프로그램의 성능을 P로 측정했을 때 경험E로 인해 성능이 향상됐다면, 이 컴퓨터 프로그램은 작업 T와 성능 측정 P에 대해 경험 E로 학습한 것이다.
                                         - 톰 미첼(Tom Mitchell, 1997)

위키백과 문서를 모두 내려받는 것 -> 많은 데이터를 갖게 되는 것 : 머신 러닝 X

### 1.2 왜 머신 러닝을 사용하는가?

- 기존 솔루션으로는 많은 수동 조정과 규칙이 필요한 문제 : 하나의 머신러닝 모델이 코드를 간단하고 더 잘 수행되도록 할 수 있습니다.
- 전통적인 방식으로는 전혀 해결 방법이 없는 복잡한 문제 : 가장 뛰어난 머신러닝 기법으로 해결 방법을 찾을 수 있습니다.
-  유동적인 환경 : 머신러닝 시스템은 새로운 데이터에 적응할 수 있습니다.
-  복잡한 문제와 대량의 데이터에서 통찰 얻기

### 1.3 머신러닝 시스템의 종류

- 사람의 감독 하에 훈련하는 것인지 그렇지 않은 것인지(지도, 비지도, 준지도, 강화 학습)
-  실시간으로 점진적인 학습을 하는지 아닌지(온라인 학습과 배치 학습)
-  단순하게 알고 있는 데이터 포인트와 새 데이터 포인트를 비교하는 것인지 아니면 훈련 데이터셋에서 과학자들처럼 패턴을 발견하여 예측 모델을 만드는지(사례 기반 학습과 모델 기반 학습)

-> 서로 배타적이지 않으며 연결 가능

#### 1.3.1  지도 학습과 비지도 학습
학습하는 동안의 감독 형태나 정보량'에 따라 분류

#### 지도학습(supervised learning)

-> 훈련데이터에 레이블 포함

- 회귀(regression) : **예측 변수**<sup>predictor variable</sup>라 부르는 **특성**<sup>featrue</sup>(주행거리, 연식, 브랜드 등)을 사용해 중고차 가격 같은 **타깃** 수치를 예측하는 것
- 분류 (Classification) : 전형적 지도 학습

가장 중요한 지도 학습 알고리즘

- k-최근접 이웃 <sup> k-Nearest Neighbors</sup>
- 선형 회귀 <sup>Linear Regression</sup>
- 로지스틱 회귀 <sup>Logistic Regression</sup>
- 서포트 벡터 머신 <sup>Support Vector Machines(SVM)</sup>
- 결정 트리 <sup>Decision Tree</sup>와 랜덤 포레스트 <sup>Random Forests</sup>
- 신경망 <sup>Neural networks</sup>

#### 비지도 학습(unsupervised learning)

-> 훈련데이터에 레이블 미포함

가장 중요한 비지도 학습 알고리즘

- 군집 <sup>clustering</sup>
 - k-평균<sup>k-Means</sup>
 - 계층 군집 분석<sup>Hierarchical Cluster Analysis</sup>(HCA)
 - 기댓값 최대화<sup>Expectation Maximization</sup>
- 시각화<sup>visualization</sup>와 차원 축소<sup>dimensionality reduction</sup>
 - 주성분 분석<sup>Principal Component Analysis</sup>(PCA)
 - 커널<sup>kernel</sup>PCA
 - 지역적 선형 임베딩<sup>Locally-Linear Embedding</sup>(LLE)
 - t-SNE<sup>t-distributed Stochastic Neighbor Embedding</sup>
- 연관 규칙 학습<sup>Assiociation rule learning</sup>
 - 어프라이어리<sup>Apriori</sup>
 -  이클렛<sup>Eclat</sup>

시각화<sup>visualization</sup>알고리즘 : 도식화 가능한 2D나 3D 표현 , 가능한 구조 유지
차원 축소<sup>dimensionality reduction</sup> : 상관관계가 있는 여러 특성을 하나로 합치는 것  ex) 주행거리, 연식 -> 차의 마모 (**특성 추출**)
이상치 탐지<sup>anomaly detection</sup> : 학습 알고리즘 주입 전 데이터셋에 이상한 값을 자동으로 제거
연관 규칙 학습<sup>association rule learning</sup> : 대량의 데이터에서 특성 간의 흥미로운 관계

#### 준지도 학습<sup>semisupervised learning</sup>

레이블이 일부만 존재

#### 강화 학습<sup>Reinforcement Learning</sup>

학습하는 시스템을 **에이전트**, 환경을 관찰해서 행동을 실행하고 보상 또는 벌점을 받습니다. 가장 큰 보상을 얻기 위해 **정책**<sup>policdy</sup>이라 부르는최상 전략을 스스로 학습합니다.

#### 1.3.2 배치 학습과 온라인 학습

입력 데이터의 스트림<sup>stream</sup>으로부터 점진적으로 학습할 수 있는 여부

**배치 학습**<sup>batch learning</sup>

- 가용한 데이터를 모두 사용해 훈련
- 제품 시스템에 적용하면 더 이상의 학습없이 실행
- 많은 컴퓨팅 자원 필요(CPU, 메모리 공간, 디스크 공간, 디스크 IO, 네트워크 IO 등)
- 자원이 제한된 시스템(예 - 스마트폰, 화성 탐사 로버)이 스스로 학습해야 할 때 많은 자원 사용하면 심각한 문제

**온라인 학습**<sup>online learning</sup>

- 데이터를 순차적으로 한 개씩 또는 미니배치<sup>mini-batch</sup>라 부르는 작은 묶음 단위로 주입
- 빠른 변화에 스스로 적응해야하는 시스템에 적합, 컴퓨팅 자원이 제한된 경우
- 메인 메모리에 들어갈 수 없는 아주 큰 데이터셋을 학습하는 시스템(**외부 메모리**<sup>out-of-core</sup> 학습)
- 전체 프로세스는 보통 오프라인, 따라서 **점진적 학습**<sup>incremental learning</sup>으로 생각
- **학습률**<sup>learning rate</sup> : 변화는 데이터에 얼마나 빠르게 적응할 것

#### 1.3.3 사례 기반 학습과 모델 기반 학습

어떻게 **일반화**되는가에 따라 분류

**사례 기반 학습**<sup>instance-based learning</sup>

- **유사도**<sup>similarity</sup>를 측정하여 새로운 데이터를 일반화

**모델 기반 학습**<sup>model-based learning</sup>

- 모델을 만들어 **예측**에 사용
 - 데이터를 분석
 - 모델 선택
 - 훈련 데이터로 모델 훈련(비용 함수<sup>cost function</sup> 최소화 하는 모델 파라미터 탐색)
 - 새로운 데이터에 모델을 적용해 예측, 잘 일반화되길 기대

### 1.4 머신러닝의 주요 도전 과제

문제점
1. 나쁜 알고리즘
2. 나쁜 데이터

#### 1.4.1 충분하지 않은 양의 훈련 데이터
#### 1.4.2 대표성 없는 훈련 데이터
- 샘플이 작으면 **샘플링 잡음**<sup>sampling noise</sup>(즉, 우연에 의한 대표성 없는 데이터)
- 샘플이 큰 경우도 추출 방법이 잘못된 경우 **샘플링 편향**<sup>sampling bias</sup>

#### 1.4.3 낮은 품질의 데이터

- 에러, 이상치<sup>outlier</sup>, 잡음
- 이상치가 명확하면 무시하거나 수동으로 잘못된 것을 고침
- 일부 특성 중 데이터가 누락된 경우 특성을 무시할지, 샘플을 무시할지, 빠진값을 채울지, 특성을 넣은 모델과 제외한 모델을 따로 훈련 시킬것인지 결정

#### 1.4.4 관련 없는 특성

- **특성 공학**<sup>feature engineering</sup> : 훈련에 사용할 좋은 특성들을 찾는 것
 - **특성 선택**<sup>feature selection</sup> : 가지고 있는 특성 중에서 훈련에 가장 유용한 특성을 선택
 - **특성 추출**<sup>feature extraction</sup> : 특성을 결합하여 더 유용한 특성을 만듬(차원 축소 알고리즘)
 -  새 특성을 만듬

#### 1.4.5 훈련 데이터 과대적합

- **과대적합**<sup>overfitting</sup> : 모델이 훈련 데이터에 너무 잘 맞지만 일반성이 떨어짐
 - 훈련 데이터에 있는 잡음의 양에 비해 모델이 너무 복잡할 때 발생
 - 파라미터 수가 적은 모델 선택, 훈련데이터에 특성수를 줄임, 모델에 제약을 가하여 단순화(**하이퍼파라미터**<sup>hyperparameter</sup> : 학습하는 동안 적용할 규제의 양, 학습 알고리즘의 파라미터)
 - 훈련 데이터를 더 많이 모음
 - 훈련 데이터의 잡음을 줄임

#### 1.4.6 훈련 데이터 과소적합

- **과소적합**<sup>underfitting</sup> : 모델이 너무 단순해서 데이터의 내재된 구조를 학습하지 못할 때
  - 파라미터가 더 많은 강력한 모델 선택
  - 더 좋은 특성 제공(특성 엔지니어링)
  -  모델의 제약을 줄임( 규제 하이퍼파라미터를 감소)

### 1.5 테스트와 검증

- **훈련 세트** 와 **테스트 세트** 로 나누어 훈련
 - **일반화 오차**<sup>generalization error</sup>(**외부 샘플 오차**<sup>out-of-sample erro</sup>) : 새로운 샘플에 대한 오류 비율
 - 훈련 오차가 낮지만 일반화 오차가 높다면 과대 적합
- **검증 세트**<sup>validation set</sup>
 - **교차 검증**<sup>cross-validation</sup> 기법
