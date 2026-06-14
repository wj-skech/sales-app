import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import re

# 1. 상단 탭 및 메인 타이틀
st.set_page_config(page_title="AI Ins-Story", layout="wide", page_icon="🚀")
st.title("🚀 AI Ins-Story")
st.markdown("대화형 AI 비서: 아래 채팅창에 원하는 질문을 하면 실전 화법과 모바일 안내장을 생성합니다.")

# 2. API 키 로직 (영구 저장 및 자동 인식)
openai_api_key = ""
if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    st.sidebar.header("⚙️ 설정")
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""
    
    api_input = st.sidebar.text_input("OpenAI API Key 입력", type="password", value=st.session_state["api_key"])
    if api_input:
        st.session_state["api_key"] = api_input
        openai_api_key = api_input

# 3. 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv('성공스토리_RAG_Ready.csv')
    return df

try:
    df_rag = load_data()
except FileNotFoundError:
    st.error("⚠️ 데이터 파일이 없습니다. GitHub에 '성공스토리_RAG_Ready.csv'가 있는지 확인해주세요.")
    st.stop()

# 4. 검색 엔진 세팅
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_rag['텍스트'].fillna("").tolist())

# 5. 채팅 내역(메모리) 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 6. 이전 채팅 내역을 화면에 렌더링
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            tab1, tab2 = st.tabs(["🗣️ 보험설계사용 실전 화법", "📱 고객 제시용 디지털 안내장"])
            with tab1:
                st.success(msg["script"])
            with tab2:
                if msg.get("html"):
                    components.html(msg["html"], height=600, scrolling=True)
                else:
                    st.warning("디지털 안내장 렌더링 결과가 없습니다.")

# 7. 하단 고정형 채팅 입력창 (Gemini 스타일)
if prompt := st.chat_input("🔍 어떤 고객을 만나시나요? 필요한 화법이나 상황을 입력하세요."):
    
    if not openai_api_key:
        st.warning("🔑 OpenAI API Key가 필요합니다. 설정 창에 입력하거나 앱 설정(Secrets)에 등록해 주세요.")
        st.stop()

    # 사용자가 입력한 내용을 즉시 화면에 표시하고 메모리에 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 응답 생성 파트
    with st.chat_message("assistant"):
        # 데이터 검색
        query_vec = vectorizer.transform([prompt])
        similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = similarity.argsort()[-3:][::-1]
        
        retrieved_chunks = []
        for idx in top_indices:
            if similarity[idx] > 0.05:
                retrieved_chunks.append(df_rag.iloc[idx]['텍스트'])

        if not retrieved_chunks:
            error_msg = "🥲 입력하신 내용과 일치하는 성공스토리 화법을 찾지 못했습니다."
            st.info(error_msg)
            st.session_state.messages.append({"role": "assistant", "script": error_msg, "html": ""})
        else:
            with st.spinner("🤖 스크립트 작성 및 모바일 안내장 디자인을 렌더링 중입니다..."):
                try:
                    client = OpenAI(api_key=openai_api_key)
                    context = "\n\n".join([f"[참고자료]: {c}" for c in retrieved_chunks])
                    
                    system_prompt = """당신은 보험업계의 전설적인 세일즈 마스터이자 프리미엄 금융 마케터입니다.
[참고자료]로 제공된 과거 '성공스토리' 원본의 강력한 현장 화법과 세련된 안내장 구성을 완벽하게 복원하고 현대화하는 것이 당신의 유일한 임무입니다.

[파트 1: 보험설계사용 실전 클로징 스크립트]
- 🚨교과서적인 설명, 모호하고 뻔한 비유는 절대 금지합니다.
- [참고자료]의 원본 문맥과 흐름(어조, 끊어읽기, 강조점)을 철저히 분석하여, 현장에서 즉시 입 밖으로 낼 수 있는 '실제 대화형 스크립트'로 작성하세요.
- 반드시 다음 4단계 세일즈 구조를 지키세요:
  1. 도입(Hook): 고객의 허를 찌르는 강렬하고 도발적인 첫 질문.
  2. 팩트 폭격(Pain Point): 구체적인 수치, 최신 법령, 실제 사례를 들어 리스크를 극대화.
  3. 해결책(Solution): 다른 곳이 아닌 '우리가 제안하는 보장'이어야만 하는 이유와 핵심 내용을 단도직입적으로 제시.
  4. 클로징(Action): 확신에 찬 목소리로 당당하게 가입(사인)을 권유.
- 말투: 현장감 넘치고 당당한 프로의 구어체 ("~하시겠습니까?", "~하셔야 합니다.")

[파트 2: 고객 제시용 프리미엄 디지털 안내장 (HTML/CSS)]
- 🚨이미지 깨짐 방지: 외부 이미지 링크(<img src...>)는 절대 사용하지 마세요.
- 대신, 최고급 금융사 리플렛처럼 CSS 요소(부드러운 linear-gradient 배경, border-radius, box-shadow)와 고해상도 이모티콘(🚨, 📊, 🛡️, 💰, 💡)을 활용하여 시각적으로 압도하는 카드형 UI를 만드세요.
- 성의 없는 줄글 요약을 금지합니다. 고객의 시선을 멈추게 하는 '카피라이팅'으로 채우세요.
- 레이아웃 필수 구조:
  1. 프리미엄 헤더: 시선을 끄는 그라데이션 배경 + 도발적인 메인 카피(큰 글씨).
  2. 자가진단 체크리스트: 고객이 스스로 읽고 찔리게 만드는 팩트 체크 항목 3가지 (체크박스 스타일).
  3. 시각적 대비 박스: [위험 방치 시 타격] vs [당사 보험업계 최고 보장의 완벽 방어막]을 대비시켜 디자인.
  4. 신뢰의 푸터: '신뢰할 수 있는 보험업계 최고의 파트너'라는 느낌을 주는 하단 마무리.
- 반드시 ```html 로 시작해서 ``` 로 끝나는 코드 블록 안에만 작성하세요.
"""
                    user_prompt = f"**고객 상황**: {prompt}\n\n{context}\n\n위 자료를 토대로 스크립트와 HTML 안내장을 도제해줘."
                    
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                        temperature=0.5
                    )
                    
                    full_response = response.choices[0].message.content
                    html_match = re.search(r'```html\n(.*?)\n```', full_response, re.DOTALL)
                    
                    if html_match:
                        html_content = html_match.group(1)
                        script_content = full_response.replace(html_match.group(0), "").strip()
                    else:
                        html_content = ""
                        script_content = full_response

                    tab1, tab2 = st.tabs(["🗣️ 보험설계사용 실전 화법", "📱 고객 제시용 디지털 안내장"])
                    with tab1:
                        st.success(script_content)
                    with tab2:
                        if html_content:
                            components.html(html_content, height=600, scrolling=True)
                        else:
                            st.warning("디지털 안내장 렌더링에 실패했습니다.")

                    # 생성된 결과를 메모리에 저장
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "script": script_content, 
                        "html": html_content
                    })

                except Exception as e:
                    st.error(f"AI 통신 오류 발생: {e}")
