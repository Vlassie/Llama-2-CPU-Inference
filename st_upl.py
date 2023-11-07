import os, time
import yaml, box
import streamlit as st
from streamlit_extras.streaming_write import write
from streamlit_extras.stateful_chat import chat, add_message
from dotenv import load_dotenv, find_dotenv
from src.llm import get_conversation_chain
from src.classes import MainVisuals

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

# Stream answer to streamlit application
def stream(response):
    for item in response['answer']:
        yield item
        time.sleep(0.025)

# def handle_question:

def main():
    # Load environment variables from .env file
    load_dotenv(find_dotenv())

    # Set shared streamlit visuals
    main_vis = MainVisuals(title="🦜💬 Chat with multiple PDFs", 
                           type='pdf',
                           path=os.path.realpath(__file__), 
                           show_sources=False)
    main_vis.render()

    # Setup the chat
    with chat(key="my_chat"):
        if not st.session_state.my_chat: # Display inital message
            add_message('assistant', "How may I assist you today?", avatar="🦜")
        if question := st.chat_input('Ask your question here'):
            # Display question
            add_message('user', question, avatar="🧑‍💻")
            # Send the question to the llm
            with st.spinner("Thinking..."):
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    main_vis.selected_model, 
                    main_vis.length, 
                    main_vis.temp, 
                    main_vis.gpu_layers, 
                    st.session_state.vectorstore)
                response = st.session_state.conversation(
                    {'question': question}
                )
            placeholder = st.empty()
            with placeholder.container():
                write(stream(response))
            placeholder.empty()
                
            # Show llm response
            add_message('assistant', response['answer'], avatar="🦜")


if __name__ == '__main__':
    main()