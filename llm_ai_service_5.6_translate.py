from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

langs = ["Korean", "Japanese", "Chinese", "English"]
left_co, cent_co, last_co = st.columns(3)

with st.sidebar:
    language = st.radio('번역을 윈하는 언어를 선택하세요.', langs)

st.markdown('### 언어 번역 서비스')
prompt = st.text_input('번역을 원하는 텍스트를 입력하세요.')

trans_template = PromptTemplate(
    input_variables=['trans'],
    template='Your task is to translate this text to ' + language + 'TEXT: {trans}'
)

memory = ConversationBufferMemory(input_key='trans', memory_key='char_history')
llm = ChatOpenAI(model_name='gpt-4-0125-preview', temperature=0.0)
trans_chain = LLMChain(llm=llm, prompt=trans_template, verbose=True, output_key='translation', memory=memory)

if st.button('번역'):
    if prompt:
        response = trans_chain({'trans': prompt})
        st.info(response['translation'])
