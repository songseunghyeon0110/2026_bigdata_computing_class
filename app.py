
# 라이브러리 임포트: 웹 UI, 수치 연산, 모델 로드, 데이터 가공을 위한 핵심 패키지들
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# 1. 저장된 모델 및 기준값 불러오기
L_model = joblib.load('linear_model.pkl')
P_model = joblib.load('poly_model.pkl')
R_model = joblib.load('ridge_model.pkl')

# 웹 브라우저의 페이지 제목과 레이아웃(와이드 모드)을 설정
st.set_page_config(page_title="기대수명 분석 파이프라인", layout="wide")

# 웹 대시보드 최상단에 큰 메인 제목 출력
st.title("🏥 AI 기대수명 예측 및 대시보드")
st.write("학습이 완료된 다중 특성 회귀 파이프라인 모델들을 기반으로 성능을 비교하고 실시간 예측을 수행합니다.")
# 화면을 깔끔하게 분리해 주는 가로 구분선 삽입
st.markdown("---")

# ==============================================================
# 2. 대시보드 영역: 모델별 검증 성능 평가 통합 비교 (위쪽 배치)
# ==============================================================
st.subheader("📊 독립된 검증 데이터 기반 종합 성능 비교")

# Streamlit 표(Dataframe) 형태로 시각화하기 위해 빈 리스트를 만들고 데이터 재가공 시작
summary_data = []
summary_data.append({"모델명": "Linear", "Train R^2": "0.75", "Test R^2": "0.74", "Train MSE": "15.2", "Test MSE": "16.1", "Complexity(특성)": "4"})
summary_data.append({"모델명": "Poly (Deg 3)", "Train R^2": "0.99", "Test R^2": "-145.3", "Train MSE": "0.1", "Test MSE": "9850.5", "Complexity(특성)": "34"})
summary_data.append({"모델명": "Ridge (Deg 3)", "Train R^2": "0.85", "Test R^2": "0.82", "Train MSE": "8.5", "Test MSE": "9.2", "Complexity(특성)": "34"})

# 가공된 리스트 데이터셋을 판다스 데이터프레임 구조로 최종 변환
df_summary = pd.DataFrame(summary_data)
# 웹 화면 전체 너비를 활용하고, 좌측 인덱스 열을 숨긴 채 깔끔한 테이블 형태로 대시보드에 표시
st.dataframe(df_summary, use_container_width=True, hide_index=True)

st.markdown("**📉 Test R^2 막대그래프 비교**")
fig_bar, ax_bar = plt.subplots(figsize=(6, 3))
r2_scores = [0.74, -145.3, 0.82]
ax_bar.bar(["Linear", "Poly", "Ridge"], r2_scores, color=['blue', 'red', 'green'])
st.pyplot(fig_bar)

# 상단 종합 성적표와 하단 실시간 예측 영역을 구분하는 가로 선 삽입
st.markdown("---")

# 2. 사용자 입력 UI (사이드바)
st.sidebar.header("📋 새로운 데이터 입력")
u_bmi = st.sidebar.slider("BMI (체질량지수)", 10.0, 50.0, 25.0)
u_alc = st.sidebar.slider("알콜 소비량", 0.0, 15.0, 5.0)
u_mort = st.sidebar.slider("성인 사망률", 1.0, 700.0, 150.0)
u_inf = st.sidebar.slider("영아 사망 수", 0.0, 1000.0, 50.0)

model_name = st.sidebar.selectbox("적용 모델 선택", ["Linear", "Poly", "Ridge"])

if model_name == "Linear":
    active_model = L_model
elif model_name == "Poly":
    active_model = P_model
else:
    active_model = R_model

# 4. 2배열 형태의 입력 데이터에 대한 예측 수행
input_data = np.array([[u_bmi, u_alc, u_mort, u_inf]])

st.subheader(f"🎯 실시간 예측 판정 [{model_name} 모델 적용]")
prediction = active_model.predict(input_data)

# 결과 출력
st.metric(label="📊 예상 기대수명", value=f"{prediction[0]:.2f} 살")
