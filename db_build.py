# =========================
#  Module: Vector DB Build
# =========================
import box
import yaml
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
import timeit
import sys
import os

from langchain.embeddings import HuggingFaceEmbeddings

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


# Build vector database
def run_db_build():
    start = timeit.default_timer()
   
    documents = []

    files = os.listdir(cfg.DATA_PATH)
    total_files = len(files)

    for index, file in enumerate(files, start=1):
        print(f"Loading... {file} - File {index}/{total_files}", end='\r')
        print(end='\x1b[2K') # clear previous print so no overlap occurs

        if file.endswith('.pdf'):
            pdf_path = './data/' + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = './data/' + file
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith('.txt'):
            text_path = './data/' + file
            loader = TextLoader(text_path)
            documents.extend(loader.load())

    print(f"Done loading all {total_files} files")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE,
                                                   chunk_overlap=cfg.CHUNK_OVERLAP)
    print("Splitting document text ...")
    texts = text_splitter.split_documents(documents)

    print("Loading embeddings ...")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    print("Building FAISS VectorStore from documents and embeddings ...")
    vectorstore = FAISS.from_documents(texts, embeddings)
    print(f"Saving database to ./{cfg.DB_FAISS_PATH}/ ...")
    vectorstore.save_local(cfg.DB_FAISS_PATH)
    end = timeit.default_timer()
    print(f"Done building database. Time to build database: {round((end - start)/60, 2)} minutes", end="\r")


if __name__ == "__main__":
    run_db_build()
