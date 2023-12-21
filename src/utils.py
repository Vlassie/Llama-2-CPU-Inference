'''
===========================================
        Module: Util functions
===========================================
'''
import box, yaml, os, sys
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from src.prompts import qa_template

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

# Function to get the list of files in a folder
def get_files_in_folder(folder_path, ignore_file):
    files = os.listdir(folder_path)
    return [file for file in files if file != ignore_file]

# Generate user input options (which model to choose)
def generate_user_input_options(model_path):
    # Check if any models exist
    if not get_files_in_folder(model_path, 'model_download.txt'):
        st.write(f"No models available in '{model_path}'")
        sys.exit()
    else: # Get the user input options and files list for the models
        files = get_files_in_folder(model_path, 'model_download.txt')
        options = [f"Option {i+1}: {file}" for i, file in enumerate(files)]
        options_str = "\n".join(options)

    return model_path, files, options_str

def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    return embeddings
 
def get_pdf_text(pdf_docs):
    documents = []
    for pdf in pdf_docs:
        loader = PyPDFLoader(os.path.join('data', pdf.name))
        documents.extend(loader.load())

    return documents

def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE,
                                                   chunk_overlap=cfg.CHUNK_OVERLAP)
    texts = text_splitter.split_documents(docs)
    return texts

def get_vectorstore(pdf_docs):
    pdf_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(pdf_text)
    embeddings = load_embeddings()
    vectorstore = FAISS.from_documents(texts=text_chunks, embedding=embeddings)
    return vectorstore

def clear_chat_history():
    st.session_state.my_chat = []
    st.session_state.memory.clear()

def reset_prompt():
    st.session_state.text_prompt.replace(st.session_state.prompt, qa_template)
    st.session_state.prompt = qa_template