'''
===========================================
        Module: Util functions
===========================================
'''
import os, sys
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import FAISS
from src.prompts import qa_template

# Function to get the list of files in a folder
def get_files_in_folder(folder_path, ignore_file):
    files = os.listdir(folder_path)
    return [file for file in files if file != ignore_file]

# Generate user input options (which model to choose)
def generate_user_input_options(directory):
    # Obtain the current directory this file is in, and create a path for /models/
    cur_dir = os.path.dirname(directory)
    folder_path = os.path.join(cur_dir, 'models')

    # Check if any models exist
    if not get_files_in_folder(folder_path, 'model_download.txt'):
        st.write(f"No models available in '{folder_path}'")
        sys.exit()
    else: # Get the user input options and files list for the models
        files = get_files_in_folder(folder_path, 'model_download.txt')

    return folder_path, files

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(pdf_docs):
    pdf_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(pdf_text)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def clear_chat_history():
    st.session_state.my_chat = []
    st.session_state.memory.clear()

def reset_prompt():
    st.session_state.text_prompt.replace(st.session_state.prompt, qa_template)
    st.session_state.prompt = qa_template