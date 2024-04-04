from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(model_name='gpt-4-0125-preview', temperature=0.0)

st.set_page_config(page_title='이메일 작성 서비스', page_icon=':robot:')
st.header('이메일 작성 서비스')

def get_email():
    input_text = st.text_area('이메일 내용을 입력하세요.', label_visibility='collapsed', placeholder='당신의 메일은...', key="input_text")
    return input_text

input_text = get_email()

query_template = """
    메일을 작성해주세요.
    아래는 이메일입니다.
    이메일: {email}
"""

prompt = PromptTemplate(input_variables=["email"], template=query_template)

st.button('*예제를 보여주세요*', type='secondary', help='봇이 작성한 메일을 확인해보세요.')
st.markdown('### 봇이 작성한 메일은:')

if input_text:
    prompt_with_email = prompt.format(email=input_text)
    formatted_email = llm.invoke(prompt_with_email).content
    st.write(formatted_email)
