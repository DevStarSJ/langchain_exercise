from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

st.set_page_config(page_title="랭체인 뭐든지 질문하세요~")
st.title("랭체인 뭐든지 질문하세요~")

def generate_response(input_text):
    response = llm.invoke(input_text)
    print(response)
    return response.content
    
with st.form("Question"):
    text = st.text_area('질문 입력:', 'What types of text models does OpenAI provide?')
    submitted = st.form_submit_button('Submit')
    st.write(generate_response(text))
