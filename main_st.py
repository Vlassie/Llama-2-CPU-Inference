import box
import timeit
import yaml
import argparse
from dotenv import find_dotenv, load_dotenv
from src.utils import setup_dbqa
import os 
import streamlit as st
import sys
# Streamlit implementation of main.py based on: https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

# Function to get the list of files in a folder
def get_files_in_folder(folder_path):
    return os.listdir(folder_path)

# Generate user input options (which model to choose)
def generate_user_input_options(folder_path):
    files = get_files_in_folder(folder_path)
    options = [f"Option {i+1}: {file}" for i, file in enumerate(files)]
    options_str = "\n".join(options)
    return options_str, files
    
def get_sources(source_docs):
    for i, doc in enumerate(source_docs):
        with st.expander(f"Source Document {i+1}"):
            st.markdown(f'Document Name: {doc.metadata["source"]}')
            st.markdown(f'Source Text: :blue[{doc.page_content}]')
            if "page" in doc.metadata: # .txt files don't have 'pages', would otherwise return an error
                st.markdown(f'Page Number: {doc.metadata["page"]}\n')

if __name__ == "__main__":
        # Obtain the current directory this file is in, and create a path for /models/
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        folder_path = os.path.join(cur_dir, 'models')

        # Check if any models exist
        if not get_files_in_folder(folder_path):
            st.write(f"No models available in '{folder_path}'")
            sys.exit()
        else: # Get the user input options and files list for the models
            user_input_options, files = generate_user_input_options(folder_path)
        
        # App title
        st.set_page_config(page_title="🦙💬 ChatDST")

        # Replicate Credentials
        with st.sidebar:
            st.title('🦙💬 ChatDST')

            st.subheader('Models and parameters')
            selected_model = st.sidebar.selectbox('Choose a model', files, key='selected_model')
            gpu_switch = st.toggle('GPU')
            if gpu_switch:
                st.write("GPU Power activated")
                gpu_layers = 50 
            else:
                gpu_layers = 0
            temperature = st.sidebar.slider('temperature', min_value=0.000, max_value=1.0, value=0.000, step=0.005, format="%0.3f",
            help='''
            The temperature controls the 'creativity' or randomness of the model.  
            *High temperature = more diverse and creative*  
            *Low temperature = more deterministic and focused*
            '''
            )
            max_length = st.sidebar.slider('max_length', min_value=32, max_value=512, value=128, step=8, 
            help="This controls the amount of tokens the model is allowed to give as a response")
            n_sources = st.sidebar.slider('n_sources', min_value=1, max_value=5, value=2, step=1,
            help='''
            This controls the amount of sources the model returns for each answer.  
            *More sources = Longer runtime!*
            '''
            )

        # Store LLM generated responses
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

        # Display or clear chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "sources" in message: 
                    get_sources(message["sources"])
                    st.write(f":orange[Time to retrieve response: {message['time']}]")

        def clear_chat_history():
            st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

####### (Fix this) Needed for some reason to refresh cache, otherwise sources won't load correctly
        refresh = st.sidebar.button('Refresh cache')         
        
        # Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot 
        def generate_llama2_response(prompt_input, chat_box):
            string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
            for dict_message in st.session_state.messages:
                if dict_message["role"] == "user":
                    string_dialogue += "User: " + dict_message["content"] + "\n\n"
                else:
                    string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

            dbqa = setup_dbqa(os.path.join(folder_path, selected_model), length=max_length, 
                              temp=temperature, n_sources=n_sources, gpu_layers=gpu_layers,
                              chat_box=chat_box)
            output = dbqa({'question': f"{string_dialogue} {prompt_input} Assistant: "})
            answer = output["answer"]

            return output, answer
        
        # User-provided prompt
        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

        # Generate a new response if last message is not from assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            start = timeit.default_timer()
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    placeholder = st.empty()
                    output, response = generate_llama2_response(prompt, chat_box=placeholder)
                    source_docs = output['source_documents']
                    full_response = ''
                    for item in response:
                        full_response += item
                        placeholder.markdown(full_response)
                    placeholder.markdown(full_response)
            end = timeit.default_timer()
            time =  f"{round((end-start)/60)} minutes" if (end-start) > 100 else f"{round(end-start)} seconds"

            message = {"role": "assistant", "content": full_response, "sources": source_docs, "time": time}
            st.session_state.messages.append(message)