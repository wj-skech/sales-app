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
                    system_prompt = """당신은 보험업계의 최상위 탑클래스 세일즈 코치이자, 트렌디한 모바일 웹 디자이너입니다.
사용자의 질문을 분석하여 다음 [2가지 파트]로 출력하세요.

[파트 1: 보험설계사용 실전 클로징 스크립트]
- 🚨절대주의🚨: 사용자가 질문한 '특정 보험 및 상황(예: 운전자보험)'에만 100% 집중하세요. 50대라고 해서 묻지도 않은 노인성 질환, 건강보험 이야기 등을 섞어 말하면 절대 안 됩니다.
- 화법의 깊이: 누구나 아는 식상한 필요성(예: 사고 나면 다친다)은 버리세요. "우회전 일시정지 위반 12대 중과실", "민식이법 스쿨존 합의금 폭탄", "경찰조사 단계 변호사 선임비용 선지급 부재 시의 가계 파산" 등 고객이 당장 두려움을 느낄 수 있는 구체적이고 치명적인 리스크를 송곳처럼 찌르세요.

[파트 2: 고객 제시용 디지털 모바일 안내장 (HTML/CSS)]
- 🚨단순한 표(Table) 위주의 밋밋한 구성을 절대 금지합니다.🚨 고객이 스마트폰으로 보는 순간 경각심을 느낄 수 있는 '시각적 스토리텔링' 디자인을 하세요.
- 반드시 아래 3가지 요소를 포함하여 CSS 코딩을 하세요:
  1. 시각적 충격 카드(Impact Card): 🚨, 💥, ⚖️, 💸 등의 이모티콘을 큼직하게 활용하여 위험 사례별로 세련된 박스(Card) 디자인을 만드세요. 경고 문구는 붉은색 계열로 시선을 강탈하게 하세요.
  2. 사고 이미지 삽입: 시각화를 위해 화면 상단이나 위험 사례 부분에 <img src="https://picsum.photos/400/200?blur=2" style="width:100%; border-radius:12px; margin-bottom:15px; box-shadow: 0 4px 6px rgba(0,0무,0,0.1);"> 와 같은 태그를 넣어 그럴듯한 배경 이미지가 들어가게 하세요.
  3. 구조화된 레이아웃: [위험 발생(사례)] -> [경제적 타격(비용)] -> [보험의 완벽한 방어막] 순서로 스크롤하며 읽기 좋게 구성하세요.
- 반드시 ```html 로 시작해서 ``` 로 끝나는 코드 블록 안에만 HTML/CSS 코드를 작성하세요.
"""
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
