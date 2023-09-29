'''
===========================================
        Module: Util functions
===========================================
'''
import box
import yaml
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from src.prompts import qa_template
from src.llm import build_llm
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.chains import ConversationalRetrievalChain

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

def set_qa_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate.from_template(qa_template) 
    return prompt

memory = ConversationBufferMemory(memory_key='chat_history', input_key='question', output_key='answer', 
                                  return_messages=True)

def build_retrieval_qa(llm, prompt, vectordb, n_sources):
    dbqa = ConversationalRetrievalChain.from_llm(llm, 
                                                 retriever=vectordb.as_retriever(search_kwargs={'k': n_sources}),
                                                 return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
                                                 condense_question_prompt=prompt, 
                                                 memory=memory,
                                                 )
    
    return dbqa

def setup_dbqa(model_path, length, temp, n_sources, gpu_layers, chat_box=None):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)
    llm = build_llm(model_path, length, temp, gpu_layers, chat_box)
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb, n_sources)

    return dbqa
