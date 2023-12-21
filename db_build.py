# =========================
#  Module: Vector DB Build
# =========================
import box, yaml, timeit, os, sys
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from src.utils import load_embeddings
import argparse
import pickle


# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

# Added to speed up ParentDocumentRetriever with FAISS, obtained from: https://github.com/langchain-ai/langchain/issues/9929 
def monkeypatch_FAISS(embeddings_model):
    from typing import Iterable, List, Optional, Any
    def _add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> List[str]:
            """Run more texts through the embeddings and add to the vectorstore.

            Args:
                texts: Iterable of strings to add to the vectorstore.
                metadatas: Optional list of metadatas associated with the texts.
                ids: Optional list of unique IDs.

            Returns:
                List of ids from adding the texts into the vectorstore.
            """
            embeddings = embeddings_model.embed_documents(texts)
            return self._FAISS__add(texts, embeddings, metadatas=metadatas, ids=ids)

    FAISS.add_texts = _add_texts

# Save object to pickle format
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

# Build vector database
def run_db_build(childparent):
    start = timeit.default_timer()
   
    # Find data folder and file to log loaded files to
    source = cfg.DATA_PATH
    log_file = cfg.LOG_FILE
    log_path = os.path.join(source, log_file)
    all_items = os.listdir(source)
    
    # Check which files are already loaded in the database (if any)
    existing_files = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as file:
            existing_files = file.read().splitlines()
    # Obtain files that aren't yet loaded
    new_files = [name for name in all_items if name not in existing_files and name != log_file]
    
    # Check how many (new) files there are
    if new_files:
        total_files = len(new_files)
    else:
        print("No (new) files available")
        sys.exit()
    
    # Start loading files
    documents = []
    
    for index, file in enumerate(new_files, start=1):
        print(f"Loading... {file} - File {index}/{total_files}", end='\r')
        print(end='\x1b[2K') # clear previous print so no overlap occurs

        if file.endswith('.pdf'):
            pdf_path = source + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = source + file
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith('.txt'):
            text_path = source + file
            loader = TextLoader(text_path, encoding="utf8")
            documents.extend(loader.load())
    print(f"Done loading all {total_files} files")
    
    print("Loading embeddings ...")
    embeddings = load_embeddings()

    # Choose whether to create regular chunks or a combination of child and parent chunks
    if childparent:
        # Initialize FAISS with necessary components: https://api.python.langchain.com/en/latest/vectorstores/langchain.vectorstores.faiss.FAISS.html
        print("Initializing base FAISS VectorStore ...") 
        texts = ["FAISS is an important library", "LangChain supports FAISS"]
        faiss = FAISS.from_texts(texts, embeddings)

        print("Setting up ParentDocumentRetriever ...") 
        monkeypatch_FAISS(embeddings)
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.PARENT_CHUNK_SIZE, chunk_overlap=cfg.PARENT_CHUNK_OVERLAP)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHILD_CHUNK_SIZE, chunk_overlap=cfg.CHILD_CHUNK_OVERLAP)
        bigchunk_store = InMemoryStore()

        retriever = ParentDocumentRetriever(
            vectorstore=faiss, 
            docstore=bigchunk_store, 
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

        print("Adding documents to retriever ...")
        retriever.add_documents(documents)
        
        print(f"Saving retriever to ./{cfg.RETRIEVER_PATH}")
        save_object(retriever, cfg.RETRIEVER_PATH)
    else:
        print("Splitting document text ...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE,
                                                    chunk_overlap=cfg.CHUNK_OVERLAP)
        texts = text_splitter.split_documents(documents)

        # log each generated chunk for debugging
        with open('chunks.txt', 'w') as file:
            prev_source = None
            for item in texts:
                # if file != prev file -> print title of new file + first chunk
                if item.metadata['source'] != prev_source:
                    file.write(f"\n{'-'*80}\n")
                    file.write(f"{' '*20}{item.metadata['source']}")
                    file.write(f"\n{'-'*80}\n")
                    file.write("%s\n" % item.page_content)
                else: # if file == prev file -> print following chunk
                    file.write("%s\n" % item.page_content)
                file.write(f"\n{'-'*50}\n")

                prev_source = item.metadata['source']

        print("Building FAISS VectorStore from documents and embeddings ...")
        vectorstore = FAISS.from_documents(texts, embeddings)

        if os.path.isfile(cfg.DB_FAISS_PATH + '/index.faiss') & os.path.isfile(cfg.DB_FAISS_PATH + '/index.pkl'):
            print(f"Loading existing database from ./{cfg.DB_FAISS_PATH}/ ...")
            local_index = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)
            print(f"Merging new and existing databases ...")
            local_index.merge_from(vectorstore)
            print(f"Saving database to ./{cfg.DB_FAISS_PATH}/ ...")
            local_index.save_local(cfg.DB_FAISS_PATH)
        else:    
            print(f"Saving database to ./{cfg.DB_FAISS_PATH}/ ...")
            vectorstore.save_local(cfg.DB_FAISS_PATH)
    end = timeit.default_timer()
    
    # Save loaded docs names to the logging file
    with open(log_path, 'a') as file:
        for name in new_files:
            file.write(name + '\n')
    
    print(f"Done building database. Time to build database: {round((end - start)/60, 2)} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--childparent',
                        action='store_true',
                        help="Choose whether to create Child and Parent chunks or just simple chunks")
    args = parser.parse_args()
    run_db_build(args.childparent)
