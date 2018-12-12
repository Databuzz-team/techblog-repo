---
title: <Machine Learning>하이퍼파라미터 튜닝
categories:
  - HyunGeun Yoon
tags:
  - Machine Learning
  - Hyperparameter
date: 2018-12-05 15:29:33
thumbnail:
---

### 1. 모델 세부 튜닝

 프로젝트 진행 시 EDA와 데이터 전처리를 하고 자신의 목적에 적합한 모델을 선택했다고 가정을 하면 그 이후에 진행되어야할 지루한 작업중에 하나가 바로 하이퍼파라미터 값들을 튜닝하는 것일 겁니다.
 수동으로 하나 하나 조정하며 모델을 돌려볼 수도 있지만 작업이 한번 끝날때마다 새로운 값으로 조정하는 것은 매우 귀찮은 일이고 시간적으로도 낭비가 많습니다. 이러한 문제점을 해결하기 위해 Scikit Learn에서 제공하는 기능을 활용해보겠습니다.

 ### 2. 하이퍼파라미터 튜닝 방법
 #### 2.1 Grid Search

우선 Grid Search 방식은 모델에 적용하고 싶은 하이퍼 파라미터 값들을 직접 지정해서 param_grid에 설정해두면 그 값들의 조합을 적용해서 모델의 성능을 평가할 수 있습니다.

<div>
<img src="/images/HyunGeun/hyperparameter_tunning/hyperparam_tuning_01.png"/>
<span style='font-size:12px; text-align:center; display:block; color: #999;'> Grid Search</span>
</div>

위의 코드는 예시를 들기 위해서 하이퍼마라미터와 모델등을 임의로 선택하였습니다. 첫번째 dictionary는 n_estimator는 [10, 20, 30], max_features값은 [2, 4], 두번째 dictionary는 다른 하이퍼 파라미터 값은 동일하고 bootstrap값을 [False]로 선택하였습니다. 교차 검증값은 2로 설정하였기 때문에 각 파라미터 조합을 2번씩 훈련하도록 코드를 작성하였습니다.

<div>
<img src="/images/HyunGeun/hyperparameter_tunning/hyperparam_tuning_02.png"/>
<span style='font-size:12px; text-align:center; display:block; color: #999;'> Grid Search result</span>
</div>

이 결과를 보면 첫번째 dictionary는 제가 선택한 n_estimators값과 max_features값을 조합하여 2 X 3 = 6개를 두번씩 평가하여 12개 두번째 dictionary는 bootstrap값을 False로 설정하여 평가하여 총 24개를 평가하였음을 볼 수 있습니다. 저는 결과를 빠르게 보기위해 파라미터의 조합의 경우를 줄이고 교차 검증도 2번만 하였지만 실제 프로젝트를 진행하는 경우에는 좀 더 많은 실험을 할 것 입니다.

<div>
<img src="/images/HyunGeun/hyperparameter_tunning/hyperparam_tuning_03.png"/>
<span style='font-size:12px; text-align:center; display:block; color: #999;'> GridSearchCV hyperparameter</span>
</div>

또한 GridSearchCV의 parameter값이 어떻게 설정되어 작동하였는지를 알 수 있습니다. 여기에서 refit이 True로 설정되어 있으면 교차 검증으로 최적의 추정기를 찾은 다음 전체 훈련 세트로 다시 훈련을 시킵니다. 데이터를 여러개로 나누어 학습을 시킨것보다 데이터의 양이 많아지므로 더 좋은 성능을 기대할 수 있습니다.

최적의 조합값이 무엇인지도 저장이 되어 있으므로 이렇게 그 값을 확인이 가능합니다.
<div>
<img src="/images/HyunGeun/hyperparameter_tunning/hyperparam_tuning_04.png"/>
<span style='font-size:12px; text-align:center; display:block; color: #999;'> Best Hyperparameter</span>
</div>

최적의 추정기 또한 확인이 가능합니다.

<div>
<img src="/images/HyunGeun/hyperparameter_tunning/hyperparam_tuning_05.png"/>
<span style='font-size:12px; text-align:center; display:block; color: #999;'> Best Estimator</span>
</div>

또한 아래와 같이 각각의 조합값의 평가 점수도 확인이 가능하기 때문에 자신에게 필요한 조합값을 선택할 수도 있습니다.

<div>
<img src="/images/HyunGeun/hyperparameter_tunning/hyperparam_tuning_06.png"/>
<span style='font-size:12px; text-align:center; display:block; color: #999;'> Cross validation score</span>
</div>

Grid Search 방법은 편리하지만 대략적으로 최적의 hyperparameter의 조합을 알고 있거나 적은 수의 조합을 실험해볼때는 적합하지만 탐색공간이 넓은 경우에는 적합한 방법이 아닙니다. 그래서 이러한 경우에는 다음에서 이야기할 Random Search를 사용하면 더욱 효과적입니다.

### 2.2 Random Search

Random Search를 사용하는 방법은 Grid Search와 거의 동일합니다. 대신 하이퍼 파라미터 값을 우리가 직접 조합 가능한 값으로 정해주는 것이 아니라 범위를 지정하면 임의의 수를 선택하여 탐색합니다.

<div>
<img src="/images/HyunGeun/hyperparameter_tunning/hyperparam_tuning_07.png"/>
<span style='font-size:12px; text-align:center; display:block; color: #999;'> Random Search</span>
</div>

위의 코드를 보면 Grid Search와 비슷한 방식으로 사용할 수 있습니다. n_iter값을 50으로 지정해 두었기때문에 임의로 50번씩 다양한 수를 선택하여 모델의 성능을 평가하고 cv는 2이기 때문에 총 100번의 탐색을 반복합니다. 최적의 조합과 추정기를 확인할 수 있는 것도 Grid Search와 동일합니다.

## 3. 결론

위에서 설명한 두가지 방법외에도 Bayesian Optimization 방법도 존재합니다. 이는 추후에 내용을 추가하겠습니다.
Grid Search와 Random Search를 이용한다는 것만으로 최적의 하이퍼파라미터 값을 찾을 수 있다고 보장해주는 것은 아닙니다. 예를 들어 우리가 지정한 조합이나 범위안에 모델의 최적 하이퍼파라미터값이 없다면 시간을 낭비하는 것일 수도 있습니다. 또한 최종적으로 결정한 값들이 과연 최적의 값인지 의문을 가질 수도 있습니다. 이는 많은 경험과 도메인 지식, 요령이 필요한 부분입니다. 실제 많은 사람들이 이러한 고민을 하고 있고 다양한 방법을 제시하고 있습니다. 예를 들어 어떤 하이퍼파라미터 값을 지정해야 할지 모를 때는 연속된 10의 거듭제곱 수로 시도를 하는 경우도 있고 필요에 따라서는 더 작은 값을 지정하기도 합니다. 그렇지만 위의 두가지 방법을 잘 활용한다면 반복된 작업을 직접 수행하면서 낭비되는 시간을 줄일 수 있고 컴퓨팅 성능이 뒷받침을 해준다면 좀 더 효율적으로 하이퍼파라미터 튜닝을 하는데 도움이 될 것입니다. 
