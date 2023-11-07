'''
===========================================
        Module: Open-source LLM Setup
===========================================
'''
import streamlit as st
from langchain.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from dotenv import find_dotenv, load_dotenv
import box
import yaml

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

def build_llm(model_path, length, temp, gpu_layers):
    # Local LlamaCpp model, automatically supports multiple model types
    llm = LlamaCpp(model_path=model_path,
                    max_tokens=length, 
                    temperature=temp,
                    n_gpu_layers=gpu_layers,
                    n_batch=128, # ! arbitrary
                    callbacks=[StreamingStdOutCallbackHandler()],
                    verbose=False, # suppresses llama_model_loader output
                    streaming=True,
                    n_ctx=2048 # ! arbitrary
                    )
    return llm

def get_conversation_chain(selected_model, length, temp, gpu_layers, vectorstore):
    
    llm = build_llm(model_path=selected_model, length=length, 
                        temp=temp, gpu_layers=gpu_layers)
    prompt = PromptTemplate.from_template(st.session_state.prompt)
    memory = st.session_state.memory

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT}),
        return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
        condense_question_prompt=prompt, 
        memory=memory
        )

    return conversation_chain