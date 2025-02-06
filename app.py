# 스트림릿으로 모델을 예측하는 앱을 만들어보자

import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import numpy as np

# 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_csv('data/healthcare.csv', index_col=0)
    return df

# 모델 로드
@st.cache_resource
def load_model():
    model = joblib.load('models/healthcare_model.pkl')
    return model

def create_charts(df):
    st.header('📊 데이터 분석')
    
    # 1. 흡연 여부에 따른 보험 청구액 분포
    st.subheader('흡연 여부에 따른 보험 청구액 분포')
    fig = px.box(df, x='Smoker', y='InsuranceClaim', color='Smoker')
    st.plotly_chart(fig)
    
    # 2. 나이와 보험 청구액의 관계
    st.subheader('나이와 보험 청구액의 관계')
    fig = px.scatter(df, x='Age', y='InsuranceClaim', color='Gender',
                    title='나이별 보험 청구액 분포',
                    labels={'InsuranceClaim': '보험 청구액', 'Age': '나이'})
    st.plotly_chart(fig)
    
    # 3. BMI 구간별 평균 보험 청구액
    st.subheader('BMI 구간별 평균 보험 청구액')
    df['BMI_Category'] = pd.cut(df['BMI'], 
                               bins=[0, 18.5, 25, 30, 100],
                               labels=['저체중', '정상', '과체중', '비만'])
    bmi_avg = df.groupby('BMI_Category', observed=True)['InsuranceClaim'].mean().reset_index()
    fig = px.bar(bmi_avg, x='BMI_Category', y='InsuranceClaim',
                 title='BMI 구간별 평균 보험 청구액',
                 labels={'InsuranceClaim': '평균 보험 청구액', 'BMI_Category': 'BMI 구간'})
    st.plotly_chart(fig)
    
    # 4. 지역별 평균 보험 청구액
    st.subheader('지역별 평균 보험 청구액')
    region_avg = df.groupby('Region', observed=True)['InsuranceClaim'].mean().reset_index()
    fig = px.pie(region_avg, values='InsuranceClaim', names='Region',
                 title='지역별 평균 보험 청구액 비율')
    st.plotly_chart(fig)
    
    # 5. 주요 통계 지표
    st.subheader('💡 주요 통계 지표')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("평균 보험 청구액", f"${df['InsuranceClaim'].mean():,.2f}")
    with col2:
        st.metric("최대 보험 청구액", f"${df['InsuranceClaim'].max():,.2f}")
    with col3:
        st.metric("최소 보험 청구액", f"${df['InsuranceClaim'].min():,.2f}")

def main():
    st.title('🏥 의료 보험 청구액 예측 서비스')
    
    # 데이터 로드
    df = load_data()
    
    # 탭 생성
    tab1, tab2 = st.tabs(["예측하기", "데이터 분석"])
    
    with tab1:
        # 시작 부분에 설명 추가
        st.markdown("""
        ### 👋 보험 청구액 예측 서비스 사용 방법
        
        1. 왼쪽 사이드바에 환자 정보를 입력해 주세요
        2. '예측하기' 버튼을 클릭하면 예상 보험 청구액을 확인할 수 있습니다
        3. 입력하신 정보는 예측에만 사용되며 저장되지 않습니다
        
        ---
        """)
        
        # 사이드바에 입력 폼 생성
        st.sidebar.header('환자 정보 입력')
        
        age = st.sidebar.number_input('나이', min_value=0, max_value=100, value=30)
        gender = st.sidebar.selectbox('성별', ['Male', 'Female'])
        bmi = st.sidebar.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
        region = st.sidebar.selectbox('지역', ['North', 'South', 'East', 'West'])
        smoker = st.sidebar.selectbox('흡연 여부', ['Yes', 'No'])
        num_visits = st.sidebar.number_input('방문 횟수', min_value=0, max_value=20, value=5)
        
        # 예측 버튼
        if st.sidebar.button('예측하기'):
            # 입력 데이터 구성
            input_data = {
                'Age': age,
                'Gender': gender,
                'BMI': bmi,
                'Region': region,
                'Smoker': smoker,
                'NumVisits': num_visits
            }
            
            # 모델 로드 및 예측
            model = load_model()
            prediction = predict_insurance_claim(input_data, model)
            
            # 결과 표시
            st.header('예측 결과')
            st.write(f'예상 보험 청구액: ${prediction:,.2f}')
            
            # 예측 결과 설명
            st.info("""
            💡 예측 결과 설명:
            - 이 예측은 환자의 기본 정보를 바탕으로 계산되었습니다.
            - 실제 청구액은 구체적인 진료 내용에 따라 달라질 수 있습니다.
            - 이 예측은 참고용으로만 사용해주세요.
            """)
    
    with tab2:
        create_charts(df)

def predict_insurance_claim(data, pipeline):
    new_df = pd.DataFrame([data])
    prediction = pipeline.predict(new_df)
    return prediction[0]

if __name__ == '__main__':
    main()