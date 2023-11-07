import box
import yaml
from dotenv import find_dotenv, load_dotenv
import os, time
import streamlit as st
from src.llm import build_llm
from langchain.agents import create_csv_agent
from src.classes import MainVisuals
from streamlit_extras.stateful_chat import chat, add_message
from streamlit_extras.streaming_write import write


# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

# Stream answer to streamlit application
def stream(response):
    for item in response:
        yield item
        time.sleep(0.025)

def main():
    # Load environment variables from .env file
    load_dotenv(find_dotenv())
    
    # Set shared streamlit visuals
    main_vis = MainVisuals(title="🦜💬 ChatCSV", 
                           type='csv', 
                           path=os.path.realpath(__file__), 
                           show_sources=False)
    main_vis.render()

    user_csv = main_vis.file
    
    if user_csv is not None:
        llm = build_llm(
                main_vis.selected_model, 
                main_vis.length, 
                main_vis.temp, 
                main_vis.gpu_layers)
        # agent = create_csv_agent(llm, user_csv, verbose=True)
        st.session_state.agent = create_csv_agent(llm, user_csv, verbose=True)

    # Setup the chat
    with chat(key="my_chat"):
        if not st.session_state.my_chat: # Display inital message
            add_message('assistant', "Ask a question about your CSV", avatar="🦜")
        if question := st.chat_input('Ask your question here'):
            # Display question
            add_message('user', question, avatar="🧑‍💻")
            # Send the question to the llm
            with st.spinner("Thinking..."):
                response = st.session_state.agent.run(question)
            
            placeholder = st.empty()
            with placeholder.container():
                write(stream(response))
            placeholder.empty()

            # Show llm response
            add_message('assistant', response, avatar="🦜")

if __name__ == "__main__": 
    main()