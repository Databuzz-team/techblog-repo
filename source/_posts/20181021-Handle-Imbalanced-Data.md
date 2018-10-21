---
title: <Machine Learning> 비대칭 데이터 문제는 어떻게 해결해야 하나?
categories:
  - Danial Nam
tags:
  - Machine Learning
date: 2018-10-21 11:23:13
thumbnail:
---

<br>
> [참고 블로그(8 Tactics to Combat Imbalanced Classes in Your Machine Learning Dataset) 링크](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)
> [블로거 링크](https://machinelearningmastery.com/)

아직 데이터 사이언스에 입문한지 오래되지는 않았지만, 개인 프로젝트로 인스타그램 데이터를 크롤링하여 작업하면서 **비대칭 데이터 문제(Imbalanced Data)** 에 부딪혔다.

생각해보면 현실세계에서 우리가 예측하고자 하는 클래스가 Uniform하게 분포되어있을 확률은 낮은 것이 당연하다.

### 비대칭 데이터란?
비대칭 데이터는 일반적으로 분류 문제에서 클래스들이 균일하게 분포하지 않은 문제를 의미한다.

간단한 예를 들자면, 100개의 과일 사진 중에 사과 사진이 90개, 귤 사진이 10개인 경우다.

이 경우라면 100개 중에 랜덤하게 뽑은 사진이 무슨 사진인지 맞춰야 한다면 사과라고 말하는 것이 가장 합리적일 것이다. 이것이 비대칭 데이터에서 일어나는 가장 큰 문제점이다.

---
이제 비대칭 문제에 대해서는 이해했을 것이다. 이제 문제를 해결하기 위해 취해야 할 전략(서두에서 링크한 블로그에서 소개한 8가지 전략)을 알아보자.

## 1. 데이터를 더 모을 수 있나?
너무 당연한 질문! 더 많은 데이터는 당연히 조금은 더 클래스 대칭적인 결과를 제공할 것이므로..

## 2. 평가 기준을 바꿔보자.
**Accuracy** 는 비대칭 문제에서는 사용하면 안되는 평가 기준이다([Accuracy Paradox](https://en.wikipedia.org/wiki/Accuracy_paradox)를 참고하자).

원문 저자는 자신의 포스트 [Classification Accuracy is Not Enough: More Performance Measures You Can Use](https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/) 에서 소개한 평가기준들을 추천했다.
- Confusion Matrix:
  예측 결과를 테이블 형태로 보여준다.
- Precision:
  Positive 클래스에 속한다고 출력한 샘플 중 실제로 Positive 클래스에 속하는 샘플 수의 비율
- Recall
  실제 Positive 클래스에 속한 샘플 중에 Positive 클래스에 속한다고 출력한 표본의 수
- F1 Score
  정밀도(Precision)과 재현율(Recall)의 가중 조화 평균

이 외에도

- Kappa(or Cohen's kappa)
  Accuracy를 정규화한 값으로 보여준다.
- ROC Curves
  ROC(Receiver Operator Characteristic) 커브는 클래스 판별 기준값의 변화에 따른 위양성률(fall-out)과 재현율(recall)의 변화를 시각화한 것이다.

## 3. 데이터셋을 Re-샘플링하자.
데이터 셋을 변형시켜서 전체 클래스의 분포를 균일하게 만드는 방법으로 두 가지 방법이 있다.
1. Over-sampling
2. Under-sampling

둘 다 사용해보는 것을 추천한다.

[비대칭 데이터 문제](https://datascienceschool.net/view-notebook/c1a8dad913f74811ae8eef5d3bedc0c3/) <- 이 링크에서 다양한 샘플링 방법을 시각화해서 설명한 자료들이 있으니 확인하자.

```bash
# 다양한 샘플링 방법을 구현한 파이썬 패키지이다.

$pip install -U imbalanced-learn
```

## 4. 가짜 데이터 샘플을 만들자.
Over sampling의 기법은 가짜 데이터를 더 생성하는 것이니 위의 방법을 좀 더 발전시킨 전략이라고 보면 되겠다.

**Naive Bayes** 알고리즘을 사용할 경우에는 생성도 가능하니 이를 이용하거나, 가장 인기있는 방법인 **SMOTE(Synthetic Minority Over-sampling Technique)** 를 사용하는 것을 추천한다.

SMOTE는 부족한 클래스의 모조 샘플을 만들어내는 것이다. 이 알고리즘은 2개 이상의 비슷한 객체들을 선택하여 거리를 재고 사이사이 새로운 데이터를 생성해나간다.

자세한 정보는 [링크](http://www.jair.org/papers/paper953.html)를 확인하자.

## 5. 다른 Algorithms을 사용해보자.
언제나 그렇듯, 자신이 가장 좋아하는 알고리즘을 모든 문제에 사용하지 않는 것을 추천한다.

**의사 결정 나무(Decision Tree)** 는 비대칭 문제에서 성능이 좋은 경우가 많다.

C4.5, C5.0, CART and Random Forest 등 다양하게 사용해보는 것을 추천.

## 6. 모델에 제한을 준다.
**Penalized classification(패널티가 있는 분류)** 는 함수를 설정하여 부족한 클래스를 분류하는 것에 오류가 일어나게 만드는 것을 의미한다. 제한사항으로 설정한 함수(패널티 함수)는 부족한 클래스를 분류하는 것에 좀 더 집중을 할 수 있게 한다.

> penalized-SVM, penalized-LDA 등 penalized 된 버젼들이 존재한다.
> 그뿐만아니라, 패널라이즈드 모델들을 위해 Framework도 존재하는데, 예를들어 Weka의 [CostSensitiveClassifier](http://weka.sourceforge.net/doc.dev/weka/classifiers/meta/CostSensitiveClassifier.html#CostSensitiveClassifier--)가 있다.

패널티 매트릭스를 만드는 것은 매우 복잡하여, 특정 알고리즘을 써야 하거나 Re-샘플링이 불가능한 경우에 사용하는 것이 좋다.

## 7. 다른 관점으로 시도하자.
추천하는 방법으로는 **Anomaly Detection**, **Change Detection** 가 있다.

- [Anomaly Detection](https://en.wikipedia.org/wiki/Anomaly_detection)
- [Change Detection](https://en.wikipedia.org/wiki/Change_detection)

---
### 결론
위의 방법들은 당연하게 시행되어야 하는 순서를 보여준다고 생각한다. 조금 허무할 수도 있지만, 결국은 데이터 **사이언티스트(Scientist)** 라는 단어가 의미하듯.. 실험적인 정신을 가지고 다양한 각도에서 도전하고 가장 좋은 결과를 내기 위해 최선을 다해야 한다는 것..

---
### Related Posts
[분류 성능 평가(데이터 사이언스 스쿨)](https://datascienceschool.net/view-notebook/731e0d2ef52c41c686ba53dcaf346f32/)
