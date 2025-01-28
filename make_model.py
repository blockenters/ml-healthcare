import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Step 1: 데이터 로드 및 전처리
# CSV 파일 로드 (파일 경로는 사용자의 환경에 맞게 수정)
df = pd.read_csv('data/healthcare.csv', index_col=0)

# 타겟 변수와 피처 분리
X = df.drop(columns=["InsuranceClaim"])
y = df["InsuranceClaim"]

# Step 2: 범주형 및 수치형 데이터 처리
categorical_features = ["Gender", "Region", "Smoker"]
numerical_features = ["Age", "BMI", "NumVisits"]

# 범주형 인코딩 및 수치형 결측치 대체
categorical_transformer = OneHotEncoder(drop="first")
numerical_transformer = SimpleImputer(strategy="mean")

# ColumnTransformer를 통해 전처리 정의
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Step 3: 파이프라인 생성
# 아래는 그리드 서치 하는 경우의 코드다.  아래의 주석들을 풀면, 그리드 서치 하는 경우의 코드가 된다.
# model = RandomForestRegressor(random_state=42, n_estimators=500)
model = LinearRegression()

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Step 4: 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: 그리드서치 설정 및 실행
# param_grid = {
#     "model__n_estimators": [50, 100, 200, 300, 400, 500],
#     "model__max_depth": [None, 10, 20, 30],
#     "model__min_samples_split": [2, 5, 10],
#     "model__min_samples_leaf": [1, 2, 4]
# }

# grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
# grid_search.fit(X_train, y_train)

# # 최적의 파라미터와 성능 확인
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_

# print("최적의 파라미터:", best_params)

# Step 6: 최적 모델로 평가
# y_pred = best_model.predict(X_test)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R^2: {r2:.2f}")

# 모델을 파일로 저장하자
import joblib
joblib.dump(pipeline, 'model/healthcare_model.pkl')


# Step 7: 새로운 데이터 예측 함수
def predict_insurance_claim(new_data, pipeline):
    """
    새로운 데이터를 입력받아 보험 청구 금액을 예측
    :param new_data: dict 형태의 입력 데이터
    :param pipeline: 학습된 파이프라인 객체
    :return: 예측된 보험 청구 금액
    """
    new_df = pd.DataFrame([new_data])
    prediction = pipeline.predict(new_df)
    return prediction[0]

# 새로운 데이터 예제
new_patient = {
    "Age": 45,
    "Gender": "Male",
    "BMI": 28.5,
    "Region": "South",
    "Smoker": "Yes",
    "NumVisits": 12
}

predicted_claim = predict_insurance_claim(new_patient, pipeline)
# predicted_claim = predict_insurance_claim(new_patient, best_model)
print(f"새로운 환자의 예측 보험 청구 금액: {predicted_claim:.2f}")
