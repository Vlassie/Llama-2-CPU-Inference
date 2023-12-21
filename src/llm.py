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
from src.prompts import system_prompt
from dotenv import find_dotenv, load_dotenv
import box
import yaml
import pickle

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

# Import retriever
with open(cfg.RETRIEVER_PATH, 'rb') as inp:
    big_chunk_retriever = pickle.load(inp)

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
                    n_ctx=2048, # ! arbitrary
                    stop=["Question", "Answer", "Helpful"]
                    )
    return llm

def get_conversation_chain(selected_model, 
                           length, 
                           temp, 
                           gpu_layers, 
                           n_sources=None, 
                           vectorstore=None,
                           memory=None,
                           prompt=None
                           ):
    
    llm = build_llm(model_path=selected_model, length=length, 
                        temp=temp, gpu_layers=gpu_layers)

    # Setup retriever
    big_chunk_retriever.search_kwargs = {'k': n_sources} # pakt hier wss k=2 child sources, die samen soms minder dan k teruggeven
    retriever = big_chunk_retriever if not vectorstore else vectorstore.as_retriever(search_kwargs={'k': n_sources})
    
    systemprompt = PromptTemplate.from_template(system_prompt)
    prompt = PromptTemplate.from_template(prompt)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
        combine_docs_chain_kwargs={'prompt': systemprompt},
        condense_question_prompt=prompt,
        memory=memory, 
        )

    return conversation_chain