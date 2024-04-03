from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback


def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    documents = FAISS.from_texts(chunks, embeddings)
    return documents

def main():
    st.title("PDF 요약하기")
    st.divider()
    
    pdf = st.file_uploader("PDF 파일 업로드", type="pdf")
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        documents = process_text(text)
        query = "업로드된 DPF 파일의 내용을 요약해주세요."
        
        if query:
            docs = documents.similarity_search(query)
            llm = ChatOpenAI(model_name='gpt-4-0125-preview', temperature=0.1)
            chain = load_qa_chain(llm, chain_type='stuff')
            
            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)
                
            st.subheader("요약 결과")
            st.write(response)
            
if __name__ == '__main__':
    main()
            