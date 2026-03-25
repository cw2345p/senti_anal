import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from kiwipiepy import Kiwi
from transformers import pipeline

# --- 1. 초기 설정 및 모델 로드 (캐싱) ---
st.set_page_config(page_title="영화 리뷰 AI 분석기", layout="wide")
kiwi = Kiwi()

@st.cache_resource
def load_all_models():
    """
    모든 분석 모델을 불러와 메모리에 로드 (앱 시작 시 1회 실행)
    """
    # [MODEL 1] TF-IDF + Logistic (Colab에서 저장한 파일)
    try:
        model_data = joblib.load('movie_sentiment_model.pkl')
        vectorizer = model_data['vectorizer']
        model_lr = model_data['model']
    except FileNotFoundError:
        st.error("⚠️ 'movie_sentiment_model.pkl' 파일을 찾을 수 없습니다. Colab에서 저장한 파일을 app.py와 같은 폴더에 넣어주세요!")
        vectorizer, model_lr = None, None

    # [MODEL 2] Deep Learning (HuggingFace Transformers)
    try:
        # 한국어 영화 리뷰(NSMC)에 최적화된 KoELECTRA 모델
        # dl_pipe = pipeline("sentiment-analysis", model="white_deer/koelectra-base-v3-discriminator-finetuned-nsmc")
        dl_pipe = pipeline("text-classification", model="matthewburke/korean_sentiment", top_k=None)
    except Exception as e:
        st.error(f"⚠️ 딥러닝 모델 로드 실패: {e}")
        dl_pipe = None
        
    return vectorizer, model_lr, dl_pipe

# 모델 불러오기
vectorizer, model_lr, dl_pipe = load_all_models()

# --- 2. 전처리 함수 ---
def preprocess_korean(text):
    if not text: return ""
    tokens = kiwi.tokenize(text)
    # 명사(NN), 동사(V), 형용사(VA) 위주로 추출하여 공백으로 연결
    return " ".join([t.form for t in tokens if t.tag.startswith(('NN', 'V', 'VA'))])

# --- 3. UI 레이아웃 및 입력란 ---
st.title("🎬 영화 리뷰 감성 분석 : 모델별 비교")
st.markdown("입력하신 리뷰에 대해 **전통적 방식(TF-IDF)**과 **최신 방식(Deep Learning)**이 어떻게 다른지 비교해보세요.")

# 사이드바: 예시 리뷰 제공
with st.sidebar:
    st.header("💡 예시 리뷰 클릭해보기")
    example_reviews = [
        "정말 최고의 명작입니다! 가슴이 벅차오르네요.",
        "돈 아까워요. 개연성도 없고 지루합니다.",
        "연기는 좋은데 스토리가 유치함",
        "나쁘지 않은데 묘하게 지루함",
        "인생 영화 등극! 보는 내내 감동 그 자체입니다."
    ]
    for i, ex in enumerate(example_reviews):
        if st.button(f"예시 {i+1}"):
            st.session_state.review_input = ex

# 입력창 (세션 스테이트 연동)
if 'review_input' not in st.session_state:
    st.session_state.review_input = ""

input_text = st.text_area("분석할 리뷰를 입력하세요:", value=st.session_state.review_input, height=120, placeholder="이 영화 정말 감동적이었어요...")

# --- 4. 분석 실행 및 결과 표시 ---
if st.button("🚀 모델별 분석 시작") and input_text:
    
    # 두 개의 컬럼으로 모델 결과 분리
    col_tfidf, col_dl = st.columns(2)

    # ---------------------------------------------------------
    # [SECTION 1] TF-IDF + Logistic (단어 빈도 기반)
    # ---------------------------------------------------------
    with col_tfidf:
        st.header("1️⃣ TF-IDF + Logistic")
        if vectorizer and model_lr:
            with st.spinner("TF-IDF 분석 중..."):
                # 1) 전처리 및 벡터화 (합치기)
                processed = preprocess_korean(input_text)
                vec = vectorizer.transform([processed])
                
                # 2) 예측 및 확신도 계산
                prob = model_lr.predict_proba(vec)[0] # [부정확률, 긍정확률]
                prob_pos = prob[1] # 긍정 확률

                # 시각화 1: 게이지 차트
                fig_g1 = go.Figure(go.Indicator(
                    mode = "gauge+number", value = prob_pos * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"결과: {'😊 긍정' if prob_pos > 0.5 else '😞 부정'}"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "white"},
                        'steps': [
                            {'range': [0, 50], 'color': "#FF4B4B"}, # 빨강
                            {'range': [50, 100], 'color': "#00CC96"} # 초록
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': prob_pos * 100
                        }
                    }
                ))
                st.plotly_chart(fig_g1, use_container_width=True)

                # 시각화 2: 판단 근거 (단어 가중치)
                st.subheader("🎯 판단 근거 (단어 가중치)")
                coeffs = model_lr.coef_[0]
                features = vectorizer.get_feature_names_out()
                nonzero = vec.nonzero()[1]
                contribs = []
                for idx in nonzero:
                    # 단어 기여도 = (TF-IDF 수치) * (로지스틱 가중치)
                    contrib = vec[0, idx] * coeffs[idx]
                    contribs.append({'단어': features[idx], '기여도': contrib})
                
                if contribs:
                    df_c = pd.DataFrame(contribs).sort_values("기여도", ascending=True)
                    df_c['감성'] = np.where(df_c['기여도'] > 0, '긍정', '부정')
                    fig_bar = px.bar(
                        df_c, x='기여도', y='단어', orientation='h',
                        color='감성',
                        color_discrete_map={'긍정': '#00CC96', '부정': '#FF4B4B'},
                        text_auto='.3f'
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("분석할 유의미한 단어가 없습니다.")
        else:
            st.warning("TF-IDF 모델을 로드할 수 없습니다.")

    # ---------------------------------------------------------
    # [SECTION 2] Deep Learning (KoELECTRA - 문맥 기반)
    # ---------------------------------------------------------
    with col_dl:
        st.header("2️⃣ Deep Learning (BERT)")
        if dl_pipe:
            with st.spinner("딥러닝 분석 중..."):
                # 1) 딥러닝 예측 (문맥 파악)
                # 모델에 따라 LABEL이 긍정/부정인지 확인 필요 (여기서는 LABEL_1을 긍정으로 가정)
                # 결과를 받을 때 바로 [0]을 붙여서 첫 번째 딕셔너리를 꺼냅니다.
                result = dl_pipe(input_text)[0] 

                # 이제 딕셔너리 접근이 가능해집니다.
                score_pos = result[0]['score'] if result[0]['label'] == 'LABEL_1' else 1 - result[0]['score']

                # 시각화 1: 게이지 차트 (색상 차별화)
                fig_g2 = go.Figure(go.Indicator(
                    mode = "gauge+number", value = score_pos * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"결과: {'😊 긍정' if score_pos > 0.5 else '😞 부정'}"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#636EFA"},
                        'steps': [
                            {'range': [0, 50], 'color': "#EF553B"}, # 빨강
                            {'range': [50, 100], 'color': "#00CC96"} # 초록
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': score_pos * 100
                        }
                    }
                ))
                st.plotly_chart(fig_g2, use_container_width=True)

                # 시각화 2: 문맥 분석 리포트
                st.subheader("🧠 문맥 분석 리포트")
                sentiment_text = "긍정" if score_pos > 0.5 else "부정"
                
                if score_pos > 0.8 or score_pos < 0.2:
                    st.success(f"딥러닝 모델은 이 문장에 대해 매우 높은 확신도로 **{sentiment_text}** 성향을 나타낸다고 분석했습니다.")
                else:
                    st.warning(f"딥러닝 모델은 이 문장이 **{sentiment_text}** 에 가깝지만, 다소 복잡하거나 애매한 뉘앙스를 포함하고 있다고 분석했습니다.")
                    
                st.info("딥러닝 모델은 단어 개별의 의미보다 **문장 전체의 흐름과 뉘앙스**를 파악합니다.")
            
        else:
            st.warning("딥러닝 모델을 로드하는 중입니다... (잠시만 기다려주세요)")

    # ---------------------------------------------------------
    # [SECTION 3] 분석 방식 차이점 요약
    # ---------------------------------------------------------
    st.markdown("---")
    st.header("3️⃣ 분석 방식 차이점 요약")
    
    diff_col1, diff_col2 = st.columns(2)
    with diff_col1:
        st.write("**TF-IDF 방식:**")
        st.caption("사전적 단어 의미에 집중합니다. 특정 단어가 포함되면 즉각 반응하지만, 반어법이나 문장 구조 변화에 약합니다.")
    with diff_col2:
        st.write("**딥러닝 방식:**")
        st.caption("문장의 '기분'을 이해합니다. 단어 하나에 집착하지 않고 문장 전체가 주는 느낌을 수치화하여 정확도가 더 높습니다.")

else:
    st.info("좌측 상단의 예시를 클릭하거나 리뷰를 직접 입력하고 '분석 시작' 버튼을 눌러주세요.")