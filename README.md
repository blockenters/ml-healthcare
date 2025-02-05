# ml-healthcare

1. 병원 진료비 예측 모델 만들기 : 리니어 리그레션
2. 모델을 예측하는 앱 만들기 : 스트림릿
3. 허깅페이스에 배포하는것은 따로 있다 : https://huggingface.co/spaces/blockenters/healthcare-app/tree/main 

---

1. Pipeline 사용 시 파라미터 지정 방법

Pipeline을 사용할 때, 각각의 단계(스텝)에 이름을 붙이게 됩니다. 예를 들어, 다음과 같은 Pipeline을 만든다고 가정해볼게요:

``` python

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 첫 번째 단계: 데이터 스케일링
    ('model', RandomForestRegressor(random_state=42, n_estimators=500))  # 두 번째 단계: 모델
])
```

이제 **GridSearchCV**로 하이퍼파라미터 튜닝을 하고 싶다면, 어떤 단계의 파라미터를 조정할지 명확히 지정해야 합니다.

scaler 단계의 파라미터를 조정하려면:

예를 들어 StandardScaler의 with_mean 파라미터를 변경하고 싶다면:

``` python
param_grid = {'scaler__with_mean': [True, False]}
```

model 단계의 파라미터를 조정하려면:
RandomForestRegressor의 n_estimators 파라미터를 변경하고 싶다면:

``` python
param_grid = {'model__n_estimators': [100, 200, 300]}
```

2. 왜 이렇게 해야 하나요?

Pipeline은 여러 개의 단계(예: 전처리, 모델링 등)를 하나로 묶어줍니다. 이때, 각 단계에 별도의 하이퍼파라미터가 있을 수 있으므로, 어떤 단계의 파라미터인지 구분하기 위해 접두사(스텝 이름__파라미터)를 붙이는 것입니다.

3. 접두사 규칙 요약

형식: '스텝이름__파라미터이름'
이중 언더스코어(__) 사용: 스텝 이름과 파라미터 이름을 구분합니다.
Pipeline을 사용하지 않으면 그냥 'n_estimators'처럼 씁니다.

4. Pipeline 없이 쓸 때

만약 Pipeline을 사용하지 않고 RandomForestRegressor만 직접 GridSearchCV에 넣는다면, 접두사 없이 이렇게 씁니다:

``` python

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30]
}
```

