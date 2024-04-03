from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from streamlit_chat import message
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
import tempfile
from langchain.document_loaders import PyPDFLoader


uploaded_file = st.sidebar.file_uploader("Upload Files", type=['pdf'])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name
        
    loader = PyPDFLoader(temp_file_path)
    data = loader.load()
    embeddings = OpenAIEmbeddings()
    vectors = FAISS.from_documents(data, embeddings)
    
    chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(model_name='gpt-4-0125-preview', temperature=0.0), retriever=vectors.as_retriever())
    
    def conversation_chat(query):
        result = chain({'question': query, 'chat_history': st.session_state['history']})
        st.session_state['history'].append((query, result['answer']))
        return result['answer']
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []
        
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ['안녕하세요 !' + uploaded_file.name + '에 대해 무엇이든 물어보세요.']
        
    if 'past' not in st.session_state:
        st.session_state['past'] = ['안녕하세요!']
        
    response_container = st.container()
    container = st.container()
    
    with container:
        with st.form(key='Conv_Question', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="PDF 파일에 대해 물어보세요.", key='input')
            submit_button = st.form_submit_button(label='Send')
        if submit_button and user_input:
            output = conversation_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
    
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style='fun-emoji', seed='Nala')
                message(st.session_state['generated'][i], key=str(i), avatar_style='bottts', seed='Fluffy')
