# 스트림릿으로 모델을 예측하는 앱을 만들어보자

import streamlit as st
import pandas as pd
import joblib

def load_model():
    """모델을 로드하는 함수."""
    return joblib.load('model/healthcare_model.pkl')

def predict_insurance_claim(model, input_data):
    """입력 데이터를 사용하여 보험 청구 금액 예측."""
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return prediction[0]

# Streamlit App
st.title("병원 진료비 예측 앱")
st.write("이 앱은 환자의 정보를 기반으로 병원 진료비를 예측합니다.")

# 사용자 입력
st.sidebar.header("환자 정보 입력")
age = st.sidebar.number_input("나이 (Age)", min_value=0, max_value=120, value=30, step=1)
gender = st.sidebar.selectbox("성별 (Gender)", ["Male", "Female"])
bmi = st.sidebar.number_input("체질량지수 (BMI)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
region = st.sidebar.selectbox("지역 (Region)", ["Northeast", "Northwest", "Southeast", "Southwest"])
smoker = st.sidebar.selectbox("흡연 여부 (Smoker)", ["Yes", "No"])
num_visits = st.sidebar.number_input("병원 방문 횟수 (NumVisits)", min_value=0, max_value=100, value=1, step=1)

# 입력 데이터 구성
input_data = {
    "Age": age,
    "Gender": gender,
    "BMI": bmi,
    "Region": region,
    "Smoker": smoker,
    "NumVisits": num_visits
}

# 모델 로드
model = load_model()

# 예측 수행
if st.button("진료비 예측하기"):
    prediction = predict_insurance_claim(model, input_data)
    st.success(f"예상 보험 청구 금액: {prediction:.2f} 단위")