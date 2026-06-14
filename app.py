import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI  # 최신 버전으로 변경
import re

# 1. 상단 탭 및 메인 타이틀 변경 (우체통 -> AI 로봇, 이름 변경)
st.set_page_config(page_title="AI Ins-Story", layout="wide", page_icon="🚀")
st.title("🚀 AI Ins-Story")
st.markdown("현장 화법과 고객 제시용 맞춤형 모바일 안내장을 동시에 생성합니다.")

# 2. 사이드바 설정
st.sidebar.header("⚙️ 설정 및 필터")
openai_api_key = st.sidebar.text_input("OpenAI API Key 입력", type="password", help="여기에 API 키를 꼭 넣어주세요!")

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

# 5. 메인 검색창 및 전송 버튼 이름 변경
with st.form("search_form"):
    query = st.text_input("🔍 어떤 고객을 만나시나요? 필요한 화법이나 상황을 입력하세요.")
    submitted = st.form_submit_button("AI 화법 생성하기")

# 6. 작동 로직
if submitted:
    if not query:
        st.warning("⚠️ 어떤 화법이 필요하신지 검색창에 내용을 입력해 주세요!")
    elif not openai_api_key:
        st.warning("🔑 왼쪽 사이드바(설정 창)에 OpenAI API Key를 먼저 입력해 주세요!")
    else:
        # 데이터 검색
        query_vec = vectorizer.transform([query])
        similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = similarity.argsort()[-3:][::-1]
        
        retrieved_chunks = []
        for idx in top_indices:
            if similarity[idx] > 0.05:
                retrieved_chunks.append(df_rag.iloc[idx]['텍스트'])

        if not retrieved_chunks:
            st.info("🥲 입력하신 내용과 일치하는 AI 화법을 찾지 못했습니다.")
        else:
            with st.spinner("🚀 스크립트 작성 및 모바일 안내장 디자인을 렌더링 중입니다..."):
                try:
                    # 최신 버전용 API 엔진 가동
                    client = OpenAI(api_key=openai_api_key)
                    
                    context = "\n\n".join([f"[참고자료]: {c}" for c in retrieved_chunks])
                    system_prompt = """당신은 보험업계의 유능한 세일즈 코치이자 수석 웹 디자이너입니다.
다음 [2가지 파트]로 나누어 완벽한 세일즈 무기를 제작하세요. 불필요한 립서비스는 철저히 배제합니다.

[파트 1: 보험설계사용 실전 클로징 스크립트]
- [참고자료]를 바탕으로 현장의 리스크를 확실하고 냉철하게 찌르는 화법을 구사하세요.
- 고객에게 직접 말하는 단호하고 설득력 있는 구어체로 작성합니다.

[파트 2: 고객 제시용 디지털 모바일 안내장 (HTML/CSS)]
- 화법 내용의 핵심(보장 비교, 리스크 수치 등)을 고객이 한눈에 이해할 수 있도록 모바일 화면 크기의 디지털 리플렛으로 만드세요.
- 반드시 ```html 로 시작해서 ``` 로 끝나는 코드 블록 안에 HTML 코드를 작성하세요.
- <style> 태그를 사용하여 카드형 디자인, 보험전문가를 상징하는 색상 포인트, 깔끔한 표(Table)와 둥근 모서리 등 세련되고 직관적인 모바일 UI를 구현하세요.
"""
                    user_prompt = f"**고객 상황**: {query}\n\n{context}\n\n위 자료를 토대로 스크립트와 HTML 안내장을 도제해줘."
                    
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

                    tab1, tab2 = st.tabs(["🗣️ 보험설계사용 실전 화법 (스크립트)", "📱 고객 제시용 디지털 안내장"])
                    with tab1:
                        st.subheader("✨ AI 추천 실전 세일즈 스크립트")
                        st.success(script_content)
                    with tab2:
                        if html_content:
                            components.html(html_content, height=650, scrolling=True)
                        else:
                            st.warning("디지털 안내장 렌더링에 실패했습니다.")
                except Exception as e:
                    st.error(f"AI 통신 오류 발생: {e}")
